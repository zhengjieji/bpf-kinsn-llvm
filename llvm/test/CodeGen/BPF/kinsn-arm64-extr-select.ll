; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=all=disable \
; RUN:   -bpf-kinsn-mode=rotate=force < %s | FileCheck %s

define dso_local i64 @rotl64(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %l = shl i64 %x, 13
  %r = lshr i64 %x, 51
  %o = or i64 %l, %r
  ret i64 %o
}

define dso_local i64 @rotl32_zext(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %t = trunc i64 %x to i32
  %l = shl i32 %t, 5
  %r = lshr i32 %t, 27
  %o = or i32 %l, %r
  %z = zext i32 %o to i64
  ret i64 %z
}

; CHECK-LABEL: rotl64:
; CHECK: kinsn_call bpf_arm64_extr_x
; CHECK-LABEL: rotl32_zext:
; CHECK: kinsn_call bpf_arm64_extr_w
; CHECK-NOT: bpf_x86_
