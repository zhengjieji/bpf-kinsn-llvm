//===-------------- BPFKinsnSelect.cpp - BPF kinsn selection --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass is the single MachineInstr-level entry point for selecting
// verifier-facing BPF instruction sequences into kinsn pseudos.  LLVM IR and
// SelectionDAG keep doing the normal canonicalization; this pass only maps
// already-canonical MachineInstr patterns to target kfunc names/payload
// schemas.
//
//===----------------------------------------------------------------------===//

#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFTargetMachine.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineLoopInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-kinsn-select"

cl::opt<bool>
    EnableBPFKinsnSelect("bpf-enable-kinsn-select", cl::Hidden, cl::init(false),
                         cl::desc("Enable BPF kinsn MachineInstr selection"));
cl::opt<bool>
    ForceBPFKinsnSelectAll("bpf-kinsn-force-all", cl::Hidden, cl::init(false),
                           cl::desc("Select every legal BPF kinsn candidate, ignoring profitability"));

STATISTIC(NumUnarySelected, "Number of unary kinsn pseudos selected");
STATISTIC(NumMovbeSelected, "Number of movbe kinsn pseudos selected");
STATISTIC(NumIndexedLoadSelected,
          "Number of indexed load kinsn pseudos selected");
STATISTIC(NumBextrSelected, "Number of BEXTR kinsn pseudos selected");
STATISTIC(NumBmi1Selected, "Number of BMI1 kinsn pseudos selected");
STATISTIC(NumRotateSelected, "Number of rotate kinsn pseudos selected");
STATISTIC(NumShdSelected, "Number of SHLD/SHRD kinsn pseudos selected");
STATISTIC(NumWideLoadSelected,
          "Number of little-endian byte-ladder loads packed");
STATISTIC(NumCandidatesSkipped,
          "Number of non-profitable or overlapping kinsn candidates skipped");

namespace {

struct UnaryPattern {
  unsigned Opcode;
  unsigned PseudoOpcode;
};

constexpr UnaryPattern UnaryPatterns[] = {
    {BPF::BSWAP16, BPF::BPF_KINSN_X86_ROLW},
    {BPF::BE16, BPF::BPF_KINSN_X86_ROLW},
    {BPF::LE16, BPF::BPF_KINSN_X86_ROLW},
    {BPF::BSWAP32, BPF::BPF_KINSN_X86_BSWAPL},
    {BPF::BE32, BPF::BPF_KINSN_X86_BSWAPL},
    {BPF::LE32, BPF::BPF_KINSN_X86_BSWAPL},
    {BPF::BSWAP64, BPF::BPF_KINSN_X86_BSWAPQ},
    {BPF::BE64, BPF::BPF_KINSN_X86_BSWAPQ},
    {BPF::LE64, BPF::BPF_KINSN_X86_BSWAPQ},
};

struct MovbePattern {
  unsigned SwapOpcode;
  unsigned LoadOpcode;
  unsigned PseudoOpcode;
};

constexpr MovbePattern MovbePatterns[] = {
    {BPF::BSWAP32, BPF::LDW, BPF::BPF_KINSN_X86_MOVBE32},
    {BPF::BE32, BPF::LDW, BPF::BPF_KINSN_X86_MOVBE32},
    {BPF::LE32, BPF::LDW, BPF::BPF_KINSN_X86_MOVBE32},
    {BPF::BSWAP64, BPF::LDD, BPF::BPF_KINSN_X86_MOVBE64},
    {BPF::BE64, BPF::LDD, BPF::BPF_KINSN_X86_MOVBE64},
    {BPF::LE64, BPF::LDD, BPF::BPF_KINSN_X86_MOVBE64},
};

struct RotatePattern {
  unsigned OrOpcode;
  unsigned LeftShiftOpcode;
  unsigned RightShiftOpcode;
  unsigned Width;
  unsigned PseudoOpcode;
};

constexpr RotatePattern RotatePatterns[] = {
    {BPF::OR_rr, BPF::SLL_ri, BPF::SRL_ri, 64, BPF::BPF_KINSN_X86_ROLQ},
    {BPF::OR_rr_32, BPF::SLL_ri_32, BPF::SRL_ri_32, 32,
     BPF::BPF_KINSN_X86_RORXL},
};

struct ShdPattern {
  unsigned OrOpcode;
  unsigned LhsShiftOpcode;
  unsigned SrcShiftOpcode;
  unsigned Width;
  unsigned PseudoOpcode;
};

constexpr ShdPattern ShdPatterns[] = {
    {BPF::OR_rr, BPF::SLL_ri, BPF::SRL_ri, 64, BPF::BPF_KINSN_X86_SHLDQ},
    {BPF::OR_rr, BPF::SRL_ri, BPF::SLL_ri, 64, BPF::BPF_KINSN_X86_SHRDQ},
    {BPF::OR_rr_32, BPF::SLL_ri_32, BPF::SRL_ri_32, 32,
     BPF::BPF_KINSN_X86_SHLDL},
    {BPF::OR_rr_32, BPF::SRL_ri_32, BPF::SLL_ri_32, 32,
     BPF::BPF_KINSN_X86_SHRDL},
};

struct Candidate {
  enum Kind {
    Unary,
    WideLoadLE,
    Movbe,
    MovbeBE,
    IndexedLoad,
    Bextr,
    Bmi1,
    Rotate,
    Shd
  } K;
  MachineInstr *Root;
  MachineInstr *Left = nullptr;
  MachineInstr *Right = nullptr;
  MachineInstr *Extra = nullptr;
  unsigned PseudoOpcode;
  Register Src;
  Register Base;
  int64_t Offset = 0;
  unsigned Shift = 0;
  int Score;
  SmallVector<MachineInstr *, 8> Erase;
};

class BPFKinsnSelect final : public MachineFunctionPass {
public:
  static char ID;

  BPFKinsnSelect() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "BPF kinsn selector"; }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineLoopInfoWrapperPass>();
    AU.addPreserved<MachineLoopInfoWrapperPass>();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!EnableBPFKinsnSelect || skipFunction(MF.getFunction()))
      return false;

    TII = MF.getSubtarget<BPFSubtarget>().getInstrInfo();
    MRI = &MF.getRegInfo();
    Loops = &getAnalysis<MachineLoopInfoWrapperPass>().getLI();
    bool LocalSubprog = isLocalSubprog(MF.getFunction());

    SmallVector<Candidate, 16> Candidates;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineInstr &MI : MBB)
        collectCandidates(MI, Candidates, LocalSubprog);
    }

    llvm::stable_sort(Candidates, [](const Candidate &A, const Candidate &B) {
      return A.Score > B.Score;
    });

    DenseSet<MachineInstr *> Used;
    bool Changed = false;
    for (Candidate &C : Candidates) {
      if (C.Score <= 0 || overlaps(C, Used)) {
        ++NumCandidatesSkipped;
        continue;
      }
      markUsed(C, Used);
      Changed |= applyCandidate(C);
    }
    return Changed;
  }

private:
  const BPFInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;
  MachineLoopInfo *Loops = nullptr;

  static bool isLocalSubprog(const Function &F) {
    StringRef Section = F.getSection();
    return F.hasLocalLinkage() || Section.empty() || Section == ".text" ||
           Section.starts_with(".text.");
  }

  int blockWeight(const MachineBasicBlock &MBB) const {
    const MachineLoop *Loop = Loops ? Loops->getLoopFor(&MBB) : nullptr;
    return Loop ? 4 + Loop->getLoopDepth() : 1;
  }

  static int score(int DefaultScore) {
    return ForceBPFKinsnSelectAll && DefaultScore <= 0 ? 1 : DefaultScore;
  }

  void collectCandidates(MachineInstr &MI, SmallVectorImpl<Candidate> &Out,
                         bool LocalSubprog) {
    // Kinsn proof sequences may consume verifier stack. In bpf2bpf callees that
    // stack is combined with the caller, so keep local-subprog rewrites off
    // until the module proof stack contract is tightened.
    if (LocalSubprog) {
      collectWideLoadLE(MI, Out, true);
      return;
    }
    collectWideLoadLE(MI, Out, false);
    collectUnary(MI, Out);
    collectMovbeBE(MI, Out);
    collectMovbe(MI, Out);
    collectIndexedLoad(MI, Out);
    collectBextr(MI, Out);
    collectBmi1(MI, Out);
    collectRotate(MI, Out);
    collectShd(MI, Out);
  }

  void collectUnary(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    for (const UnaryPattern &Pattern : UnaryPatterns) {
      if (MI.getOpcode() != Pattern.Opcode)
        continue;

      int Score = score(blockWeight(*MI.getParent()) - 2);
      Out.push_back(
          {Candidate::Unary, &MI, nullptr, nullptr, nullptr, Pattern.PseudoOpcode,
           MI.getOperand(1).getReg(), Register(), 0, 0, Score});
      return;
    }
  }

  static bool isByteLoad32(const MachineInstr *MI) {
    return MI && MI->getOpcode() == BPF::LDB32;
  }

  struct WideLoadLane {
    unsigned Byte;
    MachineInstr *Load;
  };

  static void addUniqueErase(MachineInstr *MI,
                             SmallVectorImpl<MachineInstr *> &Erase) {
    if (!llvm::is_contained(Erase, MI))
      Erase.push_back(MI);
  }

  bool matchWideLoadByte(Register Reg, MachineBasicBlock *MBB,
                         bool Use64,
                         MachineInstr *&Load,
                         SmallVectorImpl<MachineInstr *> &Erase) const {
    if (!Reg.isVirtual() || !MRI->hasOneNonDBGUse(Reg))
      return false;

    MachineInstr *MI = MRI->getVRegDef(Reg);
    if (!MI || MI->getParent() != MBB)
      return false;

    if (MI->getOpcode() == BPF::LDB32) {
      Load = MI;
      addUniqueErase(MI, Erase);
      return true;
    }

    if (!Use64 || MI->getOpcode() != BPF::SUBREG_TO_REG)
      return false;

    Register LoadReg = MI->getOperand(1).getReg();
    if (!LoadReg.isVirtual() || !MRI->hasOneNonDBGUse(LoadReg))
      return false;
    MachineInstr *LoadMI = MRI->getVRegDef(LoadReg);
    if (!isByteLoad32(LoadMI) || LoadMI->getParent() != MBB)
      return false;

    Load = LoadMI;
    addUniqueErase(MI, Erase);
    addUniqueErase(LoadMI, Erase);
    return true;
  }

  bool collectWideLoadLane(Register Reg, MachineBasicBlock *MBB,
                           bool Use64,
                           SmallVectorImpl<WideLoadLane> &Lanes,
                           SmallVectorImpl<MachineInstr *> &Erase) const {
    if (!Reg.isVirtual() || !MRI->hasOneNonDBGUse(Reg))
      return false;

    MachineInstr *MI = MRI->getVRegDef(Reg);
    if (!MI || MI->getParent() != MBB)
      return false;

    MachineInstr *Load = nullptr;
    if (matchWideLoadByte(Reg, MBB, Use64, Load, Erase)) {
      Lanes.push_back({0, Load});
      return true;
    }

    unsigned ShiftOpcode = Use64 ? BPF::SLL_ri : BPF::SLL_ri_32;
    unsigned OrOpcode = Use64 ? BPF::OR_rr : BPF::OR_rr_32;

    if (MI->getOpcode() == ShiftOpcode) {
      int64_t Shift = MI->getOperand(2).getImm();
      if (Shift <= 0 || Shift > (Use64 ? 56 : 24) || Shift % 8)
        return false;

      Register LoadReg = MI->getOperand(1).getReg();
      if (!matchWideLoadByte(LoadReg, MBB, Use64, Load, Erase))
        return false;

      Lanes.push_back({static_cast<unsigned>(Shift / 8), Load});
      addUniqueErase(MI, Erase);
      return true;
    }

    if (MI->getOpcode() == OrOpcode) {
      addUniqueErase(MI, Erase);
      return collectWideLoadLane(MI->getOperand(1).getReg(), MBB, Use64, Lanes,
                                 Erase) &&
             collectWideLoadLane(MI->getOperand(2).getReg(), MBB, Use64, Lanes,
                                 Erase);
    }

    return false;
  }

  bool validateWideLoadLanes(ArrayRef<WideLoadLane> Lanes, unsigned Width,
                             bool BigEndian, Register &Base,
                             int64_t &Offset) const {
    if (Lanes.size() != Width)
      return false;

    bool Seen[8] = {};
    bool HaveOffset = false;
    for (const WideLoadLane &Lane : Lanes) {
      if (Lane.Byte >= Width || Seen[Lane.Byte])
        return false;
      Seen[Lane.Byte] = true;

      Register LaneBase = Lane.Load->getOperand(1).getReg();
      int64_t LaneOff = Lane.Load->getOperand(2).getImm();
      unsigned ByteOff = BigEndian ? Width - 1 - Lane.Byte : Lane.Byte;
      int64_t RootOff = LaneOff - ByteOff;
      if (!HaveOffset) {
        Base = LaneBase;
        Offset = RootOff;
        HaveOffset = true;
      } else if (Base != LaneBase || Offset != RootOff) {
        return false;
      }
    }

    for (unsigned I = 0; I < Width; ++I)
      if (!Seen[I])
        return false;

    return isInt<16>(Offset) &&
           (BigEndian || Offset % static_cast<int64_t>(Width) == 0);
  }

  bool matchWideLoadTree(MachineInstr &Root, MachineInstr *TreeRoot,
                         unsigned Width, bool BigEndian, Register &Base,
                         int64_t &Offset,
                         SmallVectorImpl<MachineInstr *> &Erase) const {
    SmallVector<WideLoadLane, 8> Lanes;
    Erase.clear();

    bool Use64 = TreeRoot->getOpcode() == BPF::OR_rr;
    unsigned OrOpcode = Use64 ? BPF::OR_rr : BPF::OR_rr_32;
    if (TreeRoot->getOpcode() == OrOpcode) {
      if (!collectWideLoadLane(TreeRoot->getOperand(1).getReg(),
                               Root.getParent(), Use64, Lanes, Erase) ||
          !collectWideLoadLane(TreeRoot->getOperand(2).getReg(),
                               Root.getParent(), Use64, Lanes, Erase))
        return false;
    } else {
      return false;
    }

    if (TreeRoot != &Root)
      addUniqueErase(TreeRoot, Erase);
    return validateWideLoadLanes(Lanes, Width, BigEndian, Base, Offset);
  }

  void collectWideLoadLE(MachineInstr &MI, SmallVectorImpl<Candidate> &Out,
                         bool VerifierNative) {
    MachineInstr *TreeRoot = &MI;
    unsigned MaxWidth = 8;
    if (MI.getOpcode() == BPF::AND_ri_32) {
      int64_t Mask = MI.getOperand(2).getImm();
      if (Mask != 0xffff)
        return;
      Register OrReg = MI.getOperand(1).getReg();
      if (!OrReg.isVirtual() || !MRI->hasOneNonDBGUse(OrReg))
        return;
      TreeRoot = MRI->getVRegDef(OrReg);
      if (!TreeRoot || TreeRoot->getParent() != MI.getParent())
        return;
      MaxWidth = 2;
    } else if (MI.getOpcode() == BPF::OR_rr_32) {
      MaxWidth = 4;
    } else if (MI.getOpcode() != BPF::OR_rr) {
      return;
    }

    for (unsigned Width : {MaxWidth, 4U, 2U}) {
      if (Width > MaxWidth)
        continue;
      Register Base;
      int64_t Offset = 0;
      SmallVector<MachineInstr *, 8> Erase;
      if (!matchWideLoadTree(MI, TreeRoot, Width, false, Base, Offset, Erase))
        continue;

      unsigned LoadOpcode =
          VerifierNative ? (Width == 8 ? BPF::LDD
                            : Width == 4 ? BPF::LDW
                                         : BPF::LDH)
                         : (Width == 8 ? BPF::BPF_KINSN_X86_MOVQ
                            : Width == 4 ? BPF::BPF_KINSN_X86_MOVL
                                         : BPF::BPF_KINSN_X86_MOVZWL);
      int Score = blockWeight(*MI.getParent()) * static_cast<int>(Width + 2) - 1;
      Candidate C{Candidate::WideLoadLE, &MI, nullptr, nullptr, nullptr,
                  LoadOpcode, Register(), Base, Offset, Width, Score};
      C.Erase = std::move(Erase);
      Out.push_back(std::move(C));
      return;
    }
  }

  void collectMovbeBE(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    MachineInstr *TreeRoot = &MI;
    unsigned MaxWidth = 8;
    if (MI.getOpcode() == BPF::AND_ri_32) {
      if (MI.getOperand(2).getImm() != 0xffff)
        return;
      Register OrReg = MI.getOperand(1).getReg();
      if (!OrReg.isVirtual() || !MRI->hasOneNonDBGUse(OrReg))
        return;
      TreeRoot = MRI->getVRegDef(OrReg);
      if (!TreeRoot || TreeRoot->getParent() != MI.getParent())
        return;
      MaxWidth = 2;
    } else if (MI.getOpcode() == BPF::OR_rr_32) {
      MaxWidth = 4;
    } else if (MI.getOpcode() != BPF::OR_rr) {
      return;
    }

    for (unsigned Width : {MaxWidth, 4U, 2U}) {
      if (Width > MaxWidth)
        continue;
      Register Base;
      int64_t Offset = 0;
      SmallVector<MachineInstr *, 8> Erase;
      if (!matchWideLoadTree(MI, TreeRoot, Width, true, Base, Offset, Erase))
        continue;

      unsigned PseudoOpcode =
          Width == 8 ? BPF::BPF_KINSN_X86_MOVBE64
                     : (Width == 4 ? BPF::BPF_KINSN_X86_MOVBE32
                                   : BPF::BPF_KINSN_X86_MOVBE16);
      int Score = blockWeight(*MI.getParent()) * static_cast<int>(Width + 3);
      Candidate C{Candidate::MovbeBE, &MI, nullptr, nullptr, nullptr,
                  PseudoOpcode, Register(), Base, Offset, Width, Score};
      C.Erase = std::move(Erase);
      Out.push_back(std::move(C));
      return;
    }
  }

  void collectMovbe(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    for (const MovbePattern &Pattern : MovbePatterns) {
      if (MI.getOpcode() != Pattern.SwapOpcode)
        continue;

      Register LoadReg = MI.getOperand(1).getReg();
      if (!LoadReg.isVirtual() || !MRI->hasOneNonDBGUse(LoadReg))
        continue;

      MachineInstr *LoadMI = MRI->getVRegDef(LoadReg);
      if (!LoadMI || LoadMI->getOpcode() != Pattern.LoadOpcode ||
          LoadMI->getParent() != MI.getParent())
        continue;

      int64_t Offset = LoadMI->getOperand(2).getImm();
      if (!isInt<16>(Offset))
        continue;

      int Score = blockWeight(*MI.getParent()) * 4 - 1;
      Out.push_back({Candidate::Movbe, &MI, LoadMI, nullptr, nullptr,
                     Pattern.PseudoOpcode, Register(),
                     LoadMI->getOperand(1).getReg(), Offset, 0, Score});
      return;
    }
  }

  static unsigned indexedLoadPseudo(unsigned Opcode) {
    switch (Opcode) {
    case BPF::LDB32:
    case BPF::LDB:
      return BPF::BPF_KINSN_X86_MOVZBL;
    case BPF::LDH32:
    case BPF::LDH:
      return BPF::BPF_KINSN_X86_MOVZWL;
    case BPF::LDW32:
    case BPF::LDW:
      return BPF::BPF_KINSN_X86_MOVL;
    case BPF::LDD:
      return BPF::BPF_KINSN_X86_MOVQ;
    default:
      return 0;
    }
  }

  void collectIndexedLoad(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    unsigned PseudoOpcode = indexedLoadPseudo(MI.getOpcode());
    if (!PseudoOpcode)
      return;

    Register AddrReg = MI.getOperand(1).getReg();
    if (!AddrReg.isVirtual() || !MRI->hasOneNonDBGUse(AddrReg))
      return;

    MachineInstr *AddrMI = MRI->getVRegDef(AddrReg);
    if (!AddrMI || AddrMI->getParent() != MI.getParent() ||
        AddrMI->getOpcode() != BPF::ADD_rr)
      return;

    int64_t Offset = MI.getOperand(2).getImm();
    if (!isInt<16>(Offset))
      return;

    int Score = blockWeight(*MI.getParent()) * 3 - 1;
    Candidate C{Candidate::IndexedLoad, &MI, AddrMI, nullptr, nullptr,
                PseudoOpcode, AddrMI->getOperand(2).getReg(),
                AddrMI->getOperand(1).getReg(), Offset, 0, Score};
    Out.push_back(std::move(C));
  }

  static unsigned lowMaskWidth(uint64_t Mask) {
    if (!Mask || !isPowerOf2_64(Mask + 1))
      return 0;
    return Log2_64(Mask + 1);
  }

  void collectBextr(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    if (MI.getOpcode() != BPF::AND_ri)
      return;

    int64_t MaskImm = MI.getOperand(2).getImm();
    if (MaskImm <= 0)
      return;

    unsigned Len = lowMaskWidth(static_cast<uint64_t>(MaskImm));
    if (!Len)
      return;

    Register ShiftReg = MI.getOperand(1).getReg();
    if (!ShiftReg.isVirtual() || !MRI->hasOneNonDBGUse(ShiftReg))
      return;

    MachineInstr *ShiftMI = MRI->getVRegDef(ShiftReg);
    if (!ShiftMI || ShiftMI->getOpcode() != BPF::SRL_ri ||
        ShiftMI->getParent() != MI.getParent())
      return;

    int64_t Start = ShiftMI->getOperand(2).getImm();
    if (Start <= 0 || Start >= 64 || Start + Len > 64)
      return;

    /*
     * This first BEXTR form has to materialize the x86 control operand, so final
     * native code is "mov control; bextr" instead of "shr; and".  That is not an
     * instruction-count win, and the measured micro result is flat-to-negative.
     * Keep the recognizer in place, but require a future form with an existing
     * control register before selecting it by default.
     */
    int Score = score(-1);
    Out.push_back({Candidate::Bextr, &MI, ShiftMI, nullptr, nullptr,
                   BPF::BPF_KINSN_X86_BEXTRQ, ShiftMI->getOperand(1).getReg(),
                   Register(), static_cast<int64_t>(Len),
                   static_cast<unsigned>(Start), Score});
  }

  bool matchBmi1Operand(Register Reg, Register OtherReg, unsigned Opcode,
                        int64_t Imm, Register &Src, MachineInstr *&AuxMI) const {
    if (!Reg.isVirtual() || !MRI->hasOneNonDBGUse(Reg))
      return false;

    AuxMI = MRI->getVRegDef(Reg);
    if (!AuxMI || AuxMI->getOpcode() != Opcode)
      return false;

    if (Opcode == BPF::ADD_ri) {
      if (AuxMI->getOperand(2).getImm() != Imm)
        return false;
      Src = AuxMI->getOperand(1).getReg();
    } else {
      Src = AuxMI->getOperand(1).getReg();
    }
    return Src == OtherReg;
  }

  void collectBmi1(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    if (MI.getOpcode() != BPF::AND_rr)
      return;

    Register A = MI.getOperand(1).getReg();
    Register B = MI.getOperand(2).getReg();
    Register Src;
    MachineInstr *AuxMI;
    unsigned PseudoOpcode = 0;

    if (matchBmi1Operand(A, B, BPF::ADD_ri, -1, Src, AuxMI) ||
        matchBmi1Operand(B, A, BPF::ADD_ri, -1, Src, AuxMI)) {
      PseudoOpcode = BPF::BPF_KINSN_X86_BLSRQ;
    } else if (matchBmi1Operand(A, B, BPF::NEG_64, 0, Src, AuxMI) ||
               matchBmi1Operand(B, A, BPF::NEG_64, 0, Src, AuxMI)) {
      PseudoOpcode = BPF::BPF_KINSN_X86_BLSIQ;
    } else {
      return;
    }

    if (AuxMI->getParent() != MI.getParent())
      return;

    int Score = blockWeight(*MI.getParent()) * 3 - 1;
    Out.push_back({Candidate::Bmi1, &MI, AuxMI, nullptr, nullptr, PseudoOpcode, Src,
                   Register(), 0, 0, Score});
  }

  bool matchRotate(const MachineInstr &MI, const RotatePattern &Pattern,
                   Register &Src, unsigned &Shift, MachineInstr *&LeftShiftMI,
                   MachineInstr *&RightShiftMI) const {
    if (MI.getOpcode() != Pattern.OrOpcode)
      return false;

    Register LeftReg = MI.getOperand(1).getReg();
    Register RightReg = MI.getOperand(2).getReg();
    if (!LeftReg.isVirtual() || !RightReg.isVirtual())
      return false;

    LeftShiftMI = MRI->getVRegDef(LeftReg);
    RightShiftMI = MRI->getVRegDef(RightReg);
    if (!LeftShiftMI || !RightShiftMI ||
        LeftShiftMI->getOpcode() != Pattern.LeftShiftOpcode ||
        RightShiftMI->getOpcode() != Pattern.RightShiftOpcode)
      return false;
    if (LeftShiftMI->getParent() != MI.getParent() ||
        RightShiftMI->getParent() != MI.getParent())
      return false;

    Register LeftSrc = LeftShiftMI->getOperand(1).getReg();
    Register RightSrc = RightShiftMI->getOperand(1).getReg();
    if (LeftSrc != RightSrc)
      return false;

    int64_t LeftImm = LeftShiftMI->getOperand(2).getImm();
    int64_t RightImm = RightShiftMI->getOperand(2).getImm();
    if (LeftImm <= 0 || RightImm <= 0 ||
        static_cast<unsigned>(LeftImm + RightImm) != Pattern.Width)
      return false;

    if (!MRI->hasOneNonDBGUse(LeftReg) || !MRI->hasOneNonDBGUse(RightReg))
      return false;

    Src = LeftSrc;
    Shift = static_cast<unsigned>(LeftImm);
    return true;
  }

  void collectRotate(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    for (const RotatePattern &Pattern : RotatePatterns) {
      Register Src;
      unsigned Shift;
      MachineInstr *LeftShiftMI;
      MachineInstr *RightShiftMI;
      if (!matchRotate(MI, Pattern, Src, Shift, LeftShiftMI, RightShiftMI))
        continue;

      int Score = blockWeight(*MI.getParent()) * 3 - 1;
      Out.push_back({Candidate::Rotate, &MI, LeftShiftMI, RightShiftMI, nullptr,
                     Pattern.PseudoOpcode, Src, Register(), 0, Shift, Score});
      return;
    }
  }

  bool matchShd(const MachineInstr &MI, const ShdPattern &Pattern,
                Register &Lhs, Register &Src, unsigned &Shift,
                MachineInstr *&LhsShiftMI, MachineInstr *&SrcShiftMI) const {
    if (MI.getOpcode() != Pattern.OrOpcode)
      return false;

    Register A = MI.getOperand(1).getReg();
    Register B = MI.getOperand(2).getReg();
    if (!A.isVirtual() || !B.isVirtual())
      return false;

    LhsShiftMI = MRI->getVRegDef(A);
    SrcShiftMI = MRI->getVRegDef(B);
    if (!LhsShiftMI || !SrcShiftMI ||
        LhsShiftMI->getOpcode() != Pattern.LhsShiftOpcode ||
        SrcShiftMI->getOpcode() != Pattern.SrcShiftOpcode)
      return false;
    if (LhsShiftMI->getParent() != MI.getParent() ||
        SrcShiftMI->getParent() != MI.getParent())
      return false;
    if (!MRI->hasOneNonDBGUse(A) || !MRI->hasOneNonDBGUse(B))
      return false;

    int64_t LhsImm = LhsShiftMI->getOperand(2).getImm();
    int64_t SrcImm = SrcShiftMI->getOperand(2).getImm();
    if (LhsImm <= 0 || SrcImm <= 0 ||
        static_cast<unsigned>(LhsImm + SrcImm) != Pattern.Width)
      return false;

    Lhs = LhsShiftMI->getOperand(1).getReg();
    Src = SrcShiftMI->getOperand(1).getReg();
    if (Lhs == Src)
      return false;

    Shift = static_cast<unsigned>(LhsImm);
    return true;
  }

  void collectShd(MachineInstr &MI, SmallVectorImpl<Candidate> &Out) {
    for (const ShdPattern &Pattern : ShdPatterns) {
      Register Lhs, Src;
      unsigned Shift;
      MachineInstr *LhsShiftMI;
      MachineInstr *SrcShiftMI;
      if (!matchShd(MI, Pattern, Lhs, Src, Shift, LhsShiftMI, SrcShiftMI))
        continue;

      int Score = blockWeight(*MI.getParent()) * 3 - 1;
      Out.push_back({Candidate::Shd, &MI, LhsShiftMI, SrcShiftMI, nullptr,
                     Pattern.PseudoOpcode, Src, Lhs, 0, Shift, Score});
      return;
    }
  }

  static bool overlaps(const Candidate &C, const DenseSet<MachineInstr *> &Used) {
    if (Used.contains(C.Root) || (C.Left && Used.contains(C.Left)) ||
        (C.Right && Used.contains(C.Right)) ||
        (C.Extra && Used.contains(C.Extra)))
      return true;
    for (MachineInstr *MI : C.Erase)
      if (Used.contains(MI))
        return true;
    return false;
  }

  static void markUsed(const Candidate &C, DenseSet<MachineInstr *> &Used) {
    Used.insert(C.Root);
    if (C.Left)
      Used.insert(C.Left);
    if (C.Right)
      Used.insert(C.Right);
    if (C.Extra)
      Used.insert(C.Extra);
    for (MachineInstr *MI : C.Erase)
      Used.insert(MI);
  }

  bool applyCandidate(const Candidate &C) {
    MachineBasicBlock &MBB = *C.Root->getParent();
    if (C.K == Candidate::MovbeBE) {
      if (C.Shift == 2) {
        Register Zero = MRI->createVirtualRegister(&BPF::GPR32RegClass);
        BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(BPF::MOV_ri_32),
                Zero)
            .addImm(0);
        BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
                C.Root->getOperand(0).getReg())
            .addReg(Zero)
            .addReg(C.Base)
            .addImm(C.Offset);
      } else {
        BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
                C.Root->getOperand(0).getReg())
            .addReg(C.Base)
            .addImm(C.Offset);
      }
      C.Root->eraseFromParent();
      for (MachineInstr *MI : C.Erase)
        MI->eraseFromParent();
      ++NumMovbeSelected;
      return true;
    }

    if (C.K == Candidate::WideLoadLE) {
      if (C.PseudoOpcode == BPF::LDD || C.PseudoOpcode == BPF::LDW ||
          C.PseudoOpcode == BPF::LDH) {
        BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
                C.Root->getOperand(0).getReg())
            .addReg(C.Base)
            .addImm(C.Offset);
      } else {
        // Scale 4 is an LLVM-only no-index marker; AsmPrinter emits X86_FORM_MEM.
        BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
                C.Root->getOperand(0).getReg())
            .addReg(C.Base)
            .addReg(C.Base)
            .addImm(4)
            .addImm(C.Offset);
      }
      C.Root->eraseFromParent();
      for (MachineInstr *MI : C.Erase)
        MI->eraseFromParent();
      ++NumWideLoadSelected;
      return true;
    }

    if (C.K == Candidate::IndexedLoad) {
      BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
              C.Root->getOperand(0).getReg())
          .addReg(C.Base)
          .addReg(C.Src)
          .addImm(C.Shift)
          .addImm(C.Offset);
      C.Root->eraseFromParent();
      if (C.Left)
        C.Left->eraseFromParent();
      ++NumIndexedLoadSelected;
      return true;
    }

    if (C.K == Candidate::Bextr) {
      Register Control = MRI->createVirtualRegister(&BPF::GPRRegClass);
      BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(BPF::MOV_ri),
              Control)
          .addImm((C.Offset << 8) | C.Shift);
      BuildMI(MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
              C.Root->getOperand(0).getReg())
          .addReg(C.Src)
          .addReg(Control);
      C.Root->eraseFromParent();
      if (C.Left)
        C.Left->eraseFromParent();
      ++NumBextrSelected;
      return true;
    }

    MachineInstrBuilder Builder = BuildMI(
        MBB, C.Root, C.Root->getDebugLoc(), TII->get(C.PseudoOpcode),
        C.Root->getOperand(0).getReg());
    if (C.K == Candidate::Movbe) {
      Builder.addReg(C.Base).addImm(C.Offset);
    } else if (C.K == Candidate::Shd) {
      Builder.addReg(C.Base).addReg(C.Src).addImm(C.Shift);
    } else {
      Builder.addReg(C.Src);
    }
    if (C.K == Candidate::Rotate)
      Builder.addImm(C.Shift);
    C.Root->eraseFromParent();
    if (C.Left)
      C.Left->eraseFromParent();
    if (C.Right)
      C.Right->eraseFromParent();
    if (C.Extra)
      C.Extra->eraseFromParent();
    if (C.K == Candidate::Rotate)
      ++NumRotateSelected;
    else if (C.K == Candidate::Movbe)
      ++NumMovbeSelected;
    else if (C.K == Candidate::Bmi1)
      ++NumBmi1Selected;
    else if (C.K == Candidate::Shd)
      ++NumShdSelected;
    else
      ++NumUnarySelected;
    return true;
  }
};

} // end anonymous namespace

char BPFKinsnSelect::ID = 0;

INITIALIZE_PASS_BEGIN(BPFKinsnSelect, DEBUG_TYPE, "BPF kinsn selector", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(MachineLoopInfoWrapperPass)
INITIALIZE_PASS_END(BPFKinsnSelect, DEBUG_TYPE, "BPF kinsn selector", false,
                    false)

FunctionPass *llvm::createBPFKinsnSelectPass() { return new BPFKinsnSelect(); }
