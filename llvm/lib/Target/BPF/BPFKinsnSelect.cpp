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
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

#define DEBUG_TYPE "bpf-kinsn-select"

static cl::opt<bool>
    EnableBPFKinsnSelect("bpf-enable-kinsn-select", cl::Hidden, cl::init(false),
                         cl::desc("Enable BPF kinsn MachineInstr selection"));

STATISTIC(NumUnarySelected, "Number of unary kinsn pseudos selected");
STATISTIC(NumRotateSelected, "Number of rotate kinsn pseudos selected");
STATISTIC(NumScratchInitInserted,
          "Number of kinsn verifier scratch initializers inserted");

namespace {

struct UnaryPattern {
  unsigned Opcode;
  unsigned PseudoOpcode;
};

constexpr UnaryPattern UnaryPatterns[] = {
    {BPF::BSWAP64, BPF::BPF_KINSN_X86_BSWAPQ},
    {BPF::BE64, BPF::BPF_KINSN_X86_BSWAPQ},
    {BPF::LE64, BPF::BPF_KINSN_X86_BSWAPQ},
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
};

class BPFKinsnSelect final : public MachineFunctionPass {
public:
  static char ID;

  BPFKinsnSelect() : MachineFunctionPass(ID) {}

  StringRef getPassName() const override { return "BPF kinsn selector"; }

  bool runOnMachineFunction(MachineFunction &MF) override {
    if (!EnableBPFKinsnSelect || skipFunction(MF.getFunction()) ||
        isLocalSubprog(MF.getFunction()))
      return false;

    TII = MF.getSubtarget<BPFSubtarget>().getInstrInfo();
    MRI = &MF.getRegInfo();

    bool Changed = false;
    bool SelectedKinsn = false;
    for (MachineBasicBlock &MBB : MF) {
      for (MachineBasicBlock::iterator I = MBB.begin(), E = MBB.end();
           I != E;) {
        MachineInstr &MI = *I++;
        if (selectUnary(MI) || selectRotate(MI)) {
          Changed = true;
          SelectedKinsn = true;
        }
      }
    }
    if (SelectedKinsn)
      initializeScratchRegs(MF);
    return Changed;
  }

private:
  const BPFInstrInfo *TII = nullptr;
  MachineRegisterInfo *MRI = nullptr;

  static bool isLocalSubprog(const Function &F) {
    StringRef Section = F.getSection();
    return F.hasLocalLinkage() || Section.empty() || Section == ".text" ||
           Section.starts_with(".text.");
  }

  void initializeScratchRegs(MachineFunction &MF) {
    MachineBasicBlock &Entry = MF.front();
    MachineBasicBlock::iterator InsertPt = Entry.begin();
    const DebugLoc DL;

    for (unsigned Reg : {BPF::R6, BPF::R7, BPF::R8}) {
      BuildMI(Entry, InsertPt, DL, TII->get(BPF::MOV_ri), Reg).addImm(0);
      ++NumScratchInitInserted;
    }
  }

  bool selectUnary(MachineInstr &MI) {
    for (const UnaryPattern &Pattern : UnaryPatterns) {
      if (MI.getOpcode() != Pattern.Opcode)
        continue;

      MachineBasicBlock &MBB = *MI.getParent();
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(Pattern.PseudoOpcode),
              MI.getOperand(0).getReg())
          .addReg(MI.getOperand(1).getReg());
      MI.eraseFromParent();
      ++NumUnarySelected;
      return true;
    }
    return false;
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

  bool selectRotate(MachineInstr &MI) {
    for (const RotatePattern &Pattern : RotatePatterns) {
      Register Src;
      unsigned Shift;
      MachineInstr *LeftShiftMI;
      MachineInstr *RightShiftMI;
      if (!matchRotate(MI, Pattern, Src, Shift, LeftShiftMI, RightShiftMI))
        continue;

      MachineBasicBlock &MBB = *MI.getParent();
      BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(Pattern.PseudoOpcode),
              MI.getOperand(0).getReg())
          .addReg(Src)
          .addImm(Shift);
      MI.eraseFromParent();
      LeftShiftMI->eraseFromParent();
      RightShiftMI->eraseFromParent();
      ++NumRotateSelected;
      return true;
    }
    return false;
  }
};

} // end anonymous namespace

char BPFKinsnSelect::ID = 0;

INITIALIZE_PASS(BPFKinsnSelect, DEBUG_TYPE, "BPF kinsn selector", false, false)

FunctionPass *llvm::createBPFKinsnSelectPass() { return new BPFKinsnSelect(); }
