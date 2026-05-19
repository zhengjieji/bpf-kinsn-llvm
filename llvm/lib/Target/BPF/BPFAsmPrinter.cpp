//===-- BPFAsmPrinter.cpp - BPF LLVM assembly writer ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains a printer that converts from our internal representation
// of machine-dependent LLVM code to the BPF assembly language.
//
//===----------------------------------------------------------------------===//

#include "BPFAsmPrinter.h"
#include "BPF.h"
#include "BPFInstrInfo.h"
#include "BPFMCInstLower.h"
#include "BTFDebug.h"
#include "MCTargetDesc/BPFInstPrinter.h"
#include "TargetInfo/BPFTargetInfo.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/CodeGen/AsmPrinter.h"
#include "llvm/CodeGen/MachineConstantPool.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/MachineModuleInfo.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCExpr.h"
#include "llvm/MC/MCInst.h"
#include "llvm/MC/MCSectionELF.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/MC/MCSymbolELF.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLoweringObjectFile.h"
#include <cassert>
using namespace llvm;

#define DEBUG_TYPE "asm-printer"

namespace {
constexpr uint64_t X86FormRR = 1;
constexpr uint64_t X86FormImm = 2;
constexpr uint64_t X86FormMem = 4;

unsigned getBPFRegNo(Register Reg) {
  switch (Reg) {
  case BPF::R0:
  case BPF::W0:
    return 0;
  case BPF::R1:
  case BPF::W1:
    return 1;
  case BPF::R2:
  case BPF::W2:
    return 2;
  case BPF::R3:
  case BPF::W3:
    return 3;
  case BPF::R4:
  case BPF::W4:
    return 4;
  case BPF::R5:
  case BPF::W5:
    return 5;
  case BPF::R6:
  case BPF::W6:
    return 6;
  case BPF::R7:
  case BPF::W7:
    return 7;
  case BPF::R8:
  case BPF::W8:
    return 8;
  case BPF::R9:
  case BPF::W9:
    return 9;
  case BPF::R10:
  case BPF::W10:
    return 10;
  default:
    llvm_unreachable("unexpected BPF kinsn register");
  }
}

uint64_t packU4(uint64_t Value, unsigned Shift) {
  assert(Value < 16 && "payload nibble overflow");
  return Value << Shift;
}

uint64_t packU8(uint64_t Value, unsigned Shift) {
  assert(Value < 256 && "payload byte overflow");
  return Value << Shift;
}

uint64_t packX86RotateImmPayload(Register Dst, Register Src, uint64_t Shift,
                                 bool RequireTied) {
  if (RequireTied && Dst != Src)
    report_fatal_error("bpf_x86_rolq requires tied dst/src registers");
  return X86FormImm | packU4(getBPFRegNo(Dst), 4) |
         packU4(getBPFRegNo(Src), 8) | packU8(Shift, 12);
}

uint64_t packX86RRPayload(Register Dst, Register Src) {
  return X86FormRR | packU4(getBPFRegNo(Dst), 4) |
         packU4(getBPFRegNo(Src), 8);
}

uint64_t packX86PlainRRPayload(Register Dst, Register Src) {
  return packU4(getBPFRegNo(Dst), 0) | packU4(getBPFRegNo(Src), 4);
}

uint64_t packX86PlainRRRPayload(Register Dst, Register Src, Register Ctl) {
  return packU4(getBPFRegNo(Dst), 0) | packU4(getBPFRegNo(Src), 4) |
         packU4(getBPFRegNo(Ctl), 8);
}

uint64_t packX86MemPayload(Register Dst, Register Base, int64_t Offset) {
  if (!isInt<16>(Offset))
    report_fatal_error("x86 memory kinsn offset exceeds s16 payload");
  return X86FormMem | packU4(getBPFRegNo(Dst), 4) |
         packU4(getBPFRegNo(Base), 8) |
         (static_cast<uint64_t>(static_cast<uint16_t>(Offset)) << 12);
}

uint64_t packX86SibPayload(Register Dst, Register Base, Register Index,
                           uint64_t Scale, int64_t Offset) {
  constexpr uint64_t X86FormSib = 5;
  if (Scale > 3)
    report_fatal_error("x86 SIB scale must fit two bits");
  if (!isInt<16>(Offset))
    report_fatal_error("x86 SIB kinsn offset exceeds s16 payload");
  return X86FormSib | packU4(getBPFRegNo(Dst), 4) |
         packU4(getBPFRegNo(Base), 8) | packU4(getBPFRegNo(Index), 12) |
         (Scale << 16) |
         (static_cast<uint64_t>(static_cast<uint16_t>(Offset)) << 20);
}

uint64_t packX86MovSibPayload(const MachineInstr *MI) {
  int64_t Scale = MI->getOperand(3).getImm();
  return packX86SibPayload(MI->getOperand(0).getReg(),
                           MI->getOperand(1).getReg(),
                           MI->getOperand(2).getReg(), Scale,
                           MI->getOperand(4).getImm());
}

uint64_t packX86ShdPayload(Register Dst, Register Src, uint64_t Shift) {
  return packU4(getBPFRegNo(Dst), 0) | packU4(getBPFRegNo(Src), 4) |
         packU8(Shift, 8);
}

uint64_t packX86UnaryImmPayload(Register Dst, Register Src) {
  if (Dst != Src)
    report_fatal_error("bpf_x86_bswapq requires tied dst/src registers");
  return X86FormImm | packU4(getBPFRegNo(Dst), 4);
}

uint64_t packX86UnaryImm8Payload(Register Dst, Register Src) {
  if (Dst != Src)
    report_fatal_error("bpf_x86_rolw requires tied dst/src registers");
  return X86FormImm | packU4(getBPFRegNo(Dst), 4) | packU8(8, 8);
}

uint64_t packX86CmovPayload(Register Dst, Register Src) {
  return packU4(getBPFRegNo(Dst), 0) | packU4(getBPFRegNo(Src), 4);
}

uint64_t packX86LeaPayload(Register Dst, Register Base, Register Index) {
  constexpr uint64_t X86LeaFormReg = 1;
  return X86LeaFormReg | packU4(getBPFRegNo(Dst), 4) |
         packU4(getBPFRegNo(Base), 8) | packU4(getBPFRegNo(Index), 12) |
         (1ULL << 18) | (1ULL << 19);
}

uint64_t packX86LeaImmPayload(Register Dst, Register Base, int64_t Disp) {
  constexpr uint64_t X86LeaFormReg = 1;
  if (!isInt<32>(Disp))
    report_fatal_error("bpf_x86_lea displacement must fit s32");
  return X86LeaFormReg | packU4(getBPFRegNo(Dst), 4) |
         packU4(getBPFRegNo(Base), 8) | (1ULL << 19) |
         (static_cast<uint64_t>(static_cast<uint32_t>(Disp)) << 20);
}

void splitKinsnPayload(uint64_t Payload, unsigned &Dst, unsigned &Off,
                       unsigned &Imm) {
  Dst = Payload & 0xf;
  Off = (Payload >> 4) & 0xffff;
  Imm = (Payload >> 20) & 0xffffffff;
  assert((Payload >> 52) == 0 && "kinsn payload exceeds sidecar capacity");
}
} // namespace

BPFAsmPrinter::BPFAsmPrinter(TargetMachine &TM,
                             std::unique_ptr<MCStreamer> Streamer)
    : AsmPrinter(TM, std::move(Streamer), ID), BTF(nullptr), TM(TM) {}

BPFAsmPrinter::~BPFAsmPrinter() = default;

bool BPFAsmPrinter::doInitialization(Module &M) {
  AsmPrinter::doInitialization(M);

  // Only emit BTF when debuginfo available.
  if (MAI.doesSupportDebugInformation() && !M.debug_compile_units().empty()) {
    BTF = new BTFDebug(this);
    Handlers.push_back(std::unique_ptr<BTFDebug>(BTF));
  }

  return false;
}

const BPFTargetMachine &BPFAsmPrinter::getBTM() const {
  return static_cast<const BPFTargetMachine &>(TM);
}

bool BPFAsmPrinter::doFinalization(Module &M) {
  // Remove unused globals which are previously used for jump table.
  const BPFSubtarget *Subtarget = getBTM().getSubtargetImpl();
  if (Subtarget->hasGotox()) {
    std::vector<GlobalVariable *> Targets;
    for (GlobalVariable &Global : M.globals()) {
      if (Global.getLinkage() != GlobalValue::PrivateLinkage)
        continue;
      if (!Global.isConstant() || !Global.hasInitializer())
        continue;

      Constant *CV = dyn_cast<Constant>(Global.getInitializer());
      if (!CV)
        continue;
      ConstantArray *CA = dyn_cast<ConstantArray>(CV);
      if (!CA)
        continue;

      for (unsigned i = 1, e = CA->getNumOperands(); i != e; ++i) {
        if (!dyn_cast<BlockAddress>(CA->getOperand(i)))
          continue;
      }
      Targets.push_back(&Global);
    }

    for (GlobalVariable *GV : Targets) {
      GV->replaceAllUsesWith(PoisonValue::get(GV->getType()));
      GV->dropAllReferences();
      GV->eraseFromParent();
    }
  }

  for (GlobalObject &GO : M.global_objects()) {
    if (!GO.hasExternalWeakLinkage())
      continue;

    if (!SawTrapCall && GO.getName() == BPF_TRAP) {
      GO.eraseFromParent();
      break;
    }
  }

  return AsmPrinter::doFinalization(M);
}

namespace {

constexpr unsigned ScratchR6 = 1U << 0;
constexpr unsigned ScratchR7 = 1U << 1;
constexpr unsigned ScratchR8 = 1U << 2;
constexpr unsigned ScratchAll = ScratchR6 | ScratchR7 | ScratchR8;

static bool isKinsnScratchPhysReg(Register Reg) {
  return Reg == BPF::R6 || Reg == BPF::W6 || Reg == BPF::R7 ||
         Reg == BPF::W7 || Reg == BPF::R8 || Reg == BPF::W8;
}

static unsigned scratchBitForBPFRegNo(unsigned RegNo) {
  switch (RegNo) {
  case 6:
    return ScratchR6;
  case 7:
    return ScratchR7;
  case 8:
    return ScratchR8;
  default:
    return 0;
  }
}

static unsigned directMovbe16ScratchMask(const MachineInstr &MI) {
  unsigned Dst = getBPFRegNo(MI.getOperand(0).getReg());
  unsigned Base = getBPFRegNo(MI.getOperand(2).getReg());
  unsigned Mask = 0;
  unsigned Count = 0;

  for (unsigned RegNo : {6U, 7U, 8U}) {
    if (RegNo == Dst || RegNo == Base)
      continue;
    Mask |= scratchBitForBPFRegNo(RegNo);
    if (++Count == 2)
      return Mask;
  }
  return ScratchAll;
}

static unsigned rotateScratchMask(const MachineInstr &MI) {
  unsigned Dst = getBPFRegNo(MI.getOperand(0).getReg());
  unsigned Src = getBPFRegNo(MI.getOperand(1).getReg());

  for (unsigned RegNo : {6U, 7U, 8U})
    if (RegNo != Dst && RegNo != Src)
      return scratchBitForBPFRegNo(RegNo);
  return ScratchAll;
}

static unsigned kinsnScratchMaskForMI(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
  case BPF::BPF_KINSN_X86_BSWAPQ:
  case BPF::BPF_KINSN_X86_BSWAPL:
    return ScratchR6;
  case BPF::BPF_KINSN_X86_ROLQ:
  case BPF::BPF_KINSN_X86_RORXL:
    return rotateScratchMask(MI);
  case BPF::BPF_KINSN_X86_ROLW:
    return 0;
  case BPF::BPF_KINSN_X86_BLSIQ:
  case BPF::BPF_KINSN_X86_BLSRQ:
  case BPF::BPF_KINSN_X86_SHLDL:
  case BPF::BPF_KINSN_X86_SHLDQ:
  case BPF::BPF_KINSN_X86_SHRDL:
  case BPF::BPF_KINSN_X86_SHRDQ:
    return ScratchR6 | ScratchR7;
  case BPF::BPF_KINSN_X86_POPCNTQ:
    if (!isKinsnScratchPhysReg(MI.getOperand(0).getReg()) &&
        !isKinsnScratchPhysReg(MI.getOperand(1).getReg()))
      return ScratchR6 | ScratchR7;
    return ScratchAll;
  case BPF::BPF_KINSN_X86_MOVBE16:
    return directMovbe16ScratchMask(MI);
  case BPF::BPF_KINSN_X86_BEXTRQ:
  case BPF::BPF_KINSN_X86_CMPL:
  case BPF::BPF_KINSN_X86_CMPQ:
  case BPF::BPF_KINSN_X86_CMOVEL:
  case BPF::BPF_KINSN_X86_CMOVEQ:
  case BPF::BPF_KINSN_X86_CMOVNEL:
  case BPF::BPF_KINSN_X86_CMOVNEQ:
  case BPF::BPF_KINSN_X86_CMOVBL:
  case BPF::BPF_KINSN_X86_CMOVBQ:
    return ScratchAll;
  default:
    return 0;
  }
}

} // namespace

unsigned BPFAsmPrinter::functionKinsnScratchMask() const {
  unsigned Mask = 0;
  for (const MachineBasicBlock &MBB : *MF)
    for (const MachineInstr &MI : MBB)
      Mask |= kinsnScratchMaskForMI(MI);

  /*
   * MOV-load and direct MOVBE32/64 pseudos are intentionally absent here.
   * LLVM emits only BPF-register memory operands for them.  Direct MEM pseudos
   * need no scratch, MOVBE32/64 proof lowers to LDX+BSWAP, and SIB pseudos keep
   * dst away from address regs through their early-clobber constraint, so module
   * proof takes no-scratch fast paths.
   */
  return Mask;
}

void BPFAsmPrinter::emitScratchInit(unsigned Mask) {
  const std::pair<unsigned, Register> ScratchRegs[] = {
      {ScratchR6, BPF::R6}, {ScratchR7, BPF::R7}, {ScratchR8, BPF::R8}};
  for (auto [Bit, Reg] : ScratchRegs) {
    if (!(Mask & Bit))
      continue;
    MCInst Init;
    Init.setOpcode(BPF::MOV_ri);
    Init.addOperand(MCOperand::createReg(Reg));
    Init.addOperand(MCOperand::createImm(0));
    EmitToStreamer(*OutStreamer, Init);
  }
}

void BPFAsmPrinter::emitFunctionBodyStart() {
  AsmPrinter::emitFunctionBodyStart();
  if (unsigned Mask = functionKinsnScratchMask())
    emitScratchInit(Mask);
}

void BPFAsmPrinter::printOperand(const MachineInstr *MI, int OpNum,
                                 raw_ostream &O) {
  const MachineOperand &MO = MI->getOperand(OpNum);

  switch (MO.getType()) {
  case MachineOperand::MO_Register:
    O << BPFInstPrinter::getRegisterName(MO.getReg());
    break;

  case MachineOperand::MO_Immediate:
    O << MO.getImm();
    break;

  case MachineOperand::MO_MachineBasicBlock:
    O << *MO.getMBB()->getSymbol();
    break;

  case MachineOperand::MO_GlobalAddress:
    O << *getSymbol(MO.getGlobal());
    break;

  case MachineOperand::MO_BlockAddress: {
    MCSymbol *BA = GetBlockAddressSymbol(MO.getBlockAddress());
    O << BA->getName();
    break;
  }

  case MachineOperand::MO_ExternalSymbol:
    O << *GetExternalSymbolSymbol(MO.getSymbolName());
    break;

  case MachineOperand::MO_JumpTableIndex:
  case MachineOperand::MO_ConstantPoolIndex:
  default:
    llvm_unreachable("<unknown operand type>");
  }
}

bool BPFAsmPrinter::PrintAsmOperand(const MachineInstr *MI, unsigned OpNo,
                                    const char *ExtraCode, raw_ostream &O) {
  if (ExtraCode && ExtraCode[0])
    return AsmPrinter::PrintAsmOperand(MI, OpNo, ExtraCode, O);

  printOperand(MI, OpNo, O);
  return false;
}

bool BPFAsmPrinter::PrintAsmMemoryOperand(const MachineInstr *MI,
                                          unsigned OpNum, const char *ExtraCode,
                                          raw_ostream &O) {
  assert(OpNum + 1 < MI->getNumOperands() && "Insufficient operands");
  const MachineOperand &BaseMO = MI->getOperand(OpNum);
  const MachineOperand &OffsetMO = MI->getOperand(OpNum + 1);
  assert(BaseMO.isReg() &&
         "Unexpected base pointer for inline asm memory operand.");
  assert(OffsetMO.isImm() &&
         "Unexpected offset for inline asm memory operand.");
  int Offset = OffsetMO.getImm();

  if (ExtraCode)
    return true; // Unknown modifier.

  if (Offset < 0)
    O << "(" << BPFInstPrinter::getRegisterName(BaseMO.getReg()) << " - "
      << -Offset << ")";
  else
    O << "(" << BPFInstPrinter::getRegisterName(BaseMO.getReg()) << " + "
      << Offset << ")";

  return false;
}

void BPFAsmPrinter::emitKinsnPair(uint64_t Payload, StringRef Callee) {
  unsigned Dst, Off, Imm;
  splitKinsnPayload(Payload, Dst, Off, Imm);

  MCInst Sidecar;
  Sidecar.setOpcode(BPF::KINSN_SIDECAR);
  Sidecar.addOperand(MCOperand::createImm(Dst));
  Sidecar.addOperand(MCOperand::createImm(Off));
  Sidecar.addOperand(MCOperand::createImm(Imm));
  EmitToStreamer(*OutStreamer, Sidecar);

  MCInst Call;
  Call.setOpcode(BPF::KINSN_CALL);
  MCSymbol *Sym = GetExternalSymbolSymbol(Callee);
  Call.addOperand(
      MCOperand::createExpr(MCSymbolRefExpr::create(Sym, OutContext)));
  EmitToStreamer(*OutStreamer, Call);
}

bool BPFAsmPrinter::emitKinsnPseudo(const MachineInstr *MI) {
  switch (MI->getOpcode()) {
  case BPF::BPF_KINSN_X86_ROLQ:
    emitKinsnPair(packX86RotateImmPayload(MI->getOperand(0).getReg(),
                                           MI->getOperand(1).getReg(),
                                           MI->getOperand(2).getImm(), true),
                  "bpf_x86_rolq");
    return true;
  case BPF::BPF_KINSN_X86_ROLW:
    emitKinsnPair(packX86UnaryImm8Payload(MI->getOperand(0).getReg(),
                                          MI->getOperand(1).getReg()),
                  "bpf_x86_rolw");
    return true;
  case BPF::BPF_KINSN_X86_RORXL:
    emitKinsnPair(packX86RotateImmPayload(MI->getOperand(0).getReg(),
                                          MI->getOperand(1).getReg(),
                                          MI->getOperand(2).getImm(), false),
                  "bpf_x86_rorxl");
    return true;
  case BPF::BPF_KINSN_X86_BSWAPQ:
    emitKinsnPair(packX86UnaryImmPayload(MI->getOperand(0).getReg(),
                                         MI->getOperand(1).getReg()),
                  "bpf_x86_bswapq");
    return true;
  case BPF::BPF_KINSN_X86_BSWAPL:
    emitKinsnPair(packX86UnaryImmPayload(MI->getOperand(0).getReg(),
                                         MI->getOperand(1).getReg()),
                  "bpf_x86_bswapl");
    return true;
  case BPF::BPF_KINSN_X86_POPCNTQ:
    emitKinsnPair(packX86RRPayload(MI->getOperand(0).getReg(),
                                   MI->getOperand(1).getReg()),
                  "bpf_x86_popcntq");
    return true;
  case BPF::BPF_KINSN_X86_MOVBE16:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(2).getReg(),
                                    MI->getOperand(3).getImm()),
                  "bpf_x86_movbe16");
    return true;
  case BPF::BPF_KINSN_X86_MOVBE32:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getImm()),
                  "bpf_x86_movbe32");
    return true;
  case BPF::BPF_KINSN_X86_MOVBE64:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getImm()),
                  "bpf_x86_movbe64");
    return true;
  case BPF::BPF_KINSN_X86_MOVZBL_MEM:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getImm()),
                  "bpf_x86_movzbl");
    return true;
  case BPF::BPF_KINSN_X86_MOVZWL_MEM:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getImm()),
                  "bpf_x86_movzwl");
    return true;
  case BPF::BPF_KINSN_X86_MOVL_MEM:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getImm()),
                  "bpf_x86_movl");
    return true;
  case BPF::BPF_KINSN_X86_MOVQ_MEM:
    emitKinsnPair(packX86MemPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getImm()),
                  "bpf_x86_movq");
    return true;
  case BPF::BPF_KINSN_X86_MOVZBL:
    emitKinsnPair(packX86MovSibPayload(MI), "bpf_x86_movzbl");
    return true;
  case BPF::BPF_KINSN_X86_MOVZWL:
    emitKinsnPair(packX86MovSibPayload(MI), "bpf_x86_movzwl");
    return true;
  case BPF::BPF_KINSN_X86_MOVL:
    emitKinsnPair(packX86MovSibPayload(MI), "bpf_x86_movl");
    return true;
  case BPF::BPF_KINSN_X86_MOVQ:
    emitKinsnPair(packX86MovSibPayload(MI), "bpf_x86_movq");
    return true;
  case BPF::BPF_KINSN_X86_BEXTRQ:
    emitKinsnPair(packX86PlainRRRPayload(MI->getOperand(0).getReg(),
                                         MI->getOperand(1).getReg(),
                                         MI->getOperand(2).getReg()),
                  "bpf_x86_bextrq");
    return true;
  case BPF::BPF_KINSN_X86_BLSIQ:
    emitKinsnPair(packX86PlainRRPayload(MI->getOperand(0).getReg(),
                                        MI->getOperand(1).getReg()),
                  "bpf_x86_blsiq");
    return true;
  case BPF::BPF_KINSN_X86_BLSRQ:
    emitKinsnPair(packX86PlainRRPayload(MI->getOperand(0).getReg(),
                                        MI->getOperand(1).getReg()),
                  "bpf_x86_blsrq");
    return true;
  case BPF::BPF_KINSN_X86_LEAQ:
    emitKinsnPair(packX86LeaPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getReg()),
                  "bpf_x86_leaq");
    return true;
  case BPF::BPF_KINSN_X86_LEAL:
    emitKinsnPair(packX86LeaPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(1).getReg(),
                                    MI->getOperand(2).getReg()),
                  "bpf_x86_leal");
    return true;
  case BPF::BPF_KINSN_X86_LEAQI:
    emitKinsnPair(packX86LeaImmPayload(MI->getOperand(0).getReg(),
                                       MI->getOperand(1).getReg(),
                                       MI->getOperand(2).getImm()),
                  "bpf_x86_leaq");
    return true;
  case BPF::BPF_KINSN_X86_LEALI:
    emitKinsnPair(packX86LeaImmPayload(MI->getOperand(0).getReg(),
                                       MI->getOperand(1).getReg(),
                                       MI->getOperand(2).getImm()),
                  "bpf_x86_leal");
    return true;
  case BPF::BPF_KINSN_X86_CMPL:
    emitKinsnPair(packX86RRPayload(MI->getOperand(0).getReg(),
                                   MI->getOperand(1).getReg()),
                  "bpf_x86_cmpl");
    return true;
  case BPF::BPF_KINSN_X86_CMPQ:
    emitKinsnPair(packX86RRPayload(MI->getOperand(0).getReg(),
                                   MI->getOperand(1).getReg()),
                  "bpf_x86_cmpq");
    return true;
  case BPF::BPF_KINSN_X86_SHLDL:
    emitKinsnPair(packX86ShdPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(2).getReg(),
                                    MI->getOperand(3).getImm()),
                  "bpf_x86_shldl");
    return true;
  case BPF::BPF_KINSN_X86_SHLDQ:
    emitKinsnPair(packX86ShdPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(2).getReg(),
                                    MI->getOperand(3).getImm()),
                  "bpf_x86_shldq");
    return true;
  case BPF::BPF_KINSN_X86_SHRDL:
    emitKinsnPair(packX86ShdPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(2).getReg(),
                                    MI->getOperand(3).getImm()),
                  "bpf_x86_shrdl");
    return true;
  case BPF::BPF_KINSN_X86_SHRDQ:
    emitKinsnPair(packX86ShdPayload(MI->getOperand(0).getReg(),
                                    MI->getOperand(2).getReg(),
                                    MI->getOperand(3).getImm()),
                  "bpf_x86_shrdq");
    return true;
  case BPF::BPF_KINSN_X86_CMOVEL:
    emitKinsnPair(packX86CmovPayload(MI->getOperand(0).getReg(),
                                     MI->getOperand(2).getReg()),
                  "bpf_x86_cmovel");
    return true;
  case BPF::BPF_KINSN_X86_CMOVEQ:
    emitKinsnPair(packX86CmovPayload(MI->getOperand(0).getReg(),
                                     MI->getOperand(2).getReg()),
                  "bpf_x86_cmoveq");
    return true;
  case BPF::BPF_KINSN_X86_CMOVNEL:
    emitKinsnPair(packX86CmovPayload(MI->getOperand(0).getReg(),
                                     MI->getOperand(2).getReg()),
                  "bpf_x86_cmovnel");
    return true;
  case BPF::BPF_KINSN_X86_CMOVNEQ:
    emitKinsnPair(packX86CmovPayload(MI->getOperand(0).getReg(),
                                     MI->getOperand(2).getReg()),
                  "bpf_x86_cmovneq");
    return true;
  case BPF::BPF_KINSN_X86_CMOVBL:
    emitKinsnPair(packX86CmovPayload(MI->getOperand(0).getReg(),
                                     MI->getOperand(2).getReg()),
                  "bpf_x86_cmovbl");
    return true;
  case BPF::BPF_KINSN_X86_CMOVBQ:
    emitKinsnPair(packX86CmovPayload(MI->getOperand(0).getReg(),
                                     MI->getOperand(2).getReg()),
                  "bpf_x86_cmovbq");
    return true;
  case BPF::BPF_KINSN_ARM64_EXTR_X:
    llvm_unreachable("arm64 kinsn pseudo reached BPF x86 asm printer");
  default:
    return false;
  }
}

void BPFAsmPrinter::emitInstruction(const MachineInstr *MI) {
  if (MI->isCall()) {
    for (const MachineOperand &Op : MI->operands()) {
      if (Op.isGlobal()) {
        if (const GlobalValue *GV = Op.getGlobal())
          if (GV->getName() == BPF_TRAP)
            SawTrapCall = true;
      }
    }
  }

  if (emitKinsnPseudo(MI))
    return;

  BPF_MC::verifyInstructionPredicates(MI->getOpcode(),
                                      getSubtargetInfo().getFeatureBits());

  MCInst TmpInst;

  if (!BTF || !BTF->InstLower(MI, TmpInst)) {
    BPFMCInstLower MCInstLowering(OutContext, *this);
    MCInstLowering.Lower(MI, TmpInst);
  }
  EmitToStreamer(*OutStreamer, TmpInst);
}

void BPFAsmPrinter::emitFunctionBodyEnd() {
  // Emit .bpf_cleanup section with a flat table of
  // (call_site, landing_pad) pairs.
  const std::vector<LandingPadInfo> &LandingPads = MF->getLandingPads();
  if (LandingPads.empty())
    return;

  MCContext &Ctx = OutContext;
  auto *CleanupSec =
      Ctx.getELFSection(".bpf_cleanup", ELF::SHT_PROGBITS, ELF::SHF_ALLOC);
  OutStreamer->switchSection(CleanupSec);

  const auto &TypeInfos = MF->getTypeInfos();
  const Function &F = MF->getFunction();
  LLVMContext &LLVMCtx = F.getContext();

  // Each landing pad has BeginLabels/EndLabels marking the invoke
  // call sites that unwind to it.
  for (const LandingPadInfo &LP : LandingPads) {
    // BPF treats all landing pads as catch-all: the kernel redirects to
    // the landing pad regardless of exception type. Reject type-specific
    // catches and filters which would silently misbehave.
    for (int TId : LP.TypeIds) {
      if (TId > 0 && TypeInfos[TId - 1] != nullptr) {
        LLVMCtx.diagnose(DiagnosticInfoUnsupported(
            F, "BPF does not support type-specific exception catches yet"));
        return;
      }
      if (TId < 0) {
        LLVMCtx.diagnose(DiagnosticInfoUnsupported(
            F, "BPF does not support exception filters yet"));
        return;
      }
    }

    MCSymbol *LPLabel = LP.LandingPadLabel;
    if (!LPLabel)
      continue;
    for (unsigned i = 0, e = LP.BeginLabels.size(); i != e; ++i) {
      MCSymbol *Begin = LP.BeginLabels[i];
      MCSymbol *End = LP.EndLabels[i];

      // Each entry is 3 x 4 bytes: begin, end, landing_pad.
      // The invoke region [begin, end) may include argument setup
      // before the call. The runtime checks begin <= PC < end.
      OutStreamer->emitSymbolValue(Begin, 4);
      OutStreamer->emitSymbolValue(End, 4);
      OutStreamer->emitSymbolValue(LPLabel, 4);
    }
  }

  // Switch back to the function's section.
  OutStreamer->switchSection(MF->getSection());
}

MCSymbol *BPFAsmPrinter::getJTPublicSymbol(unsigned JTI) {
  SmallString<60> Name;
  raw_svector_ostream(Name)
      << "BPF.JT." << MF->getFunctionNumber() << '.' << JTI;
  MCSymbol *S = OutContext.getOrCreateSymbol(Name);
  if (auto *ES = static_cast<MCSymbolELF *>(S)) {
    ES->setBinding(ELF::STB_GLOBAL);
    ES->setType(ELF::STT_OBJECT);
  }
  return S;
}

void BPFAsmPrinter::emitJumpTableInfo() {
  const MachineJumpTableInfo *MJTI = MF->getJumpTableInfo();
  if (!MJTI)
    return;

  const std::vector<MachineJumpTableEntry> &JT = MJTI->getJumpTables();
  if (JT.empty())
    return;

  const TargetLoweringObjectFile &TLOF = getObjFileLowering();
  const Function &F = MF->getFunction();

  MCSection *Sec = OutStreamer->getCurrentSectionOnly();
  MCSymbol *SecStart = Sec->getBeginSymbol();

  MCSection *JTS = TLOF.getSectionForJumpTable(F, TM);
  assert(MJTI->getEntryKind() == MachineJumpTableInfo::EK_BlockAddress);
  unsigned EntrySize = MJTI->getEntrySize(getDataLayout());
  OutStreamer->switchSection(JTS);
  for (unsigned JTI = 0; JTI < JT.size(); JTI++) {
    ArrayRef<MachineBasicBlock *> JTBBs = JT[JTI].MBBs;
    if (JTBBs.empty())
      continue;

    MCSymbol *JTStart = getJTPublicSymbol(JTI);
    OutStreamer->emitLabel(JTStart);
    for (const MachineBasicBlock *MBB : JTBBs) {
      const MCExpr *Diff = MCBinaryExpr::createSub(
          MCSymbolRefExpr::create(MBB->getSymbol(), OutContext),
          MCSymbolRefExpr::create(SecStart, OutContext), OutContext);
      OutStreamer->emitValue(Diff, EntrySize);
    }
    const MCExpr *JTSize =
        MCConstantExpr::create(JTBBs.size() * EntrySize, OutContext);
    OutStreamer->emitELFSize(JTStart, JTSize);
  }
}

char BPFAsmPrinter::ID = 0;

INITIALIZE_PASS(BPFAsmPrinter, "bpf-asm-printer", "BPF Assembly Printer", false,
                false)

// Force static initialization.
extern "C" LLVM_ABI LLVM_EXTERNAL_VISIBILITY void
LLVMInitializeBPFAsmPrinter() {
  RegisterAsmPrinter<BPFAsmPrinter> X(getTheBPFleTarget());
  RegisterAsmPrinter<BPFAsmPrinter> Y(getTheBPFbeTarget());
  RegisterAsmPrinter<BPFAsmPrinter> Z(getTheBPFTarget());
}
