; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-mode=unary=force < %s | FileCheck %s --check-prefix=X86
; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=unary=force < %s | FileCheck %s --check-prefix=ARM64

define dso_local i64 @bswap64(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %0 = tail call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %0
}

; X86: kinsn_sidecar
; X86: kinsn_call
; X86: bpf_x86_bswapq

; ARM64-NOT: bpf_x86_bswapq
; ARM64: kinsn_sidecar
; ARM64: kinsn_call
; ARM64: bpf_arm64_rev_x

declare i64 @llvm.bswap.i64(i64) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
