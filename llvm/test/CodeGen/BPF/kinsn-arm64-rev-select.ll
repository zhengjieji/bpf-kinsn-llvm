; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=unary=force < %s | FileCheck %s

define dso_local i64 @bswap16_zext(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %t = trunc i64 %x to i16
  %b = tail call i16 @llvm.bswap.i16(i16 %t)
  %z = zext i16 %b to i64
  ret i64 %z
}

define dso_local i64 @bswap32_zext(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %t = trunc i64 %x to i32
  %b = tail call i32 @llvm.bswap.i32(i32 %t)
  %z = zext i32 %b to i64
  ret i64 %z
}

define dso_local i64 @bswap64(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %b = tail call i64 @llvm.bswap.i64(i64 %x)
  ret i64 %b
}

; CHECK-LABEL: bswap16_zext:
; CHECK: kinsn_call bpf_arm64_rev16_w
; CHECK-LABEL: bswap32_zext:
; CHECK: kinsn_call bpf_arm64_rev_w
; CHECK-LABEL: bswap64:
; CHECK: kinsn_call bpf_arm64_rev_x
; CHECK-NOT: bpf_x86_

declare i16 @llvm.bswap.i16(i16) #0
declare i32 @llvm.bswap.i32(i32) #0
declare i64 @llvm.bswap.i64(i64) #0

attributes #0 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
