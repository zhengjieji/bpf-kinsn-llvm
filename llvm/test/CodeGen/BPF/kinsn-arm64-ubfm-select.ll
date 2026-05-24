; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=all=disable \
; RUN:   -bpf-kinsn-mode=bextr=force < %s | FileCheck %s

define dso_local i64 @ubfm_shift_mask(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %s = lshr i64 %x, 6
  %o = and i64 %s, 31
  ret i64 %o
}

define dso_local i64 @ubfm_start_zero(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %o = and i64 %x, 63
  ret i64 %o
}

define dso_local i64 @too_wide(i64 noundef %x) local_unnamed_addr section "xdp" {
entry:
  %s = lshr i64 %x, 4
  %o = and i64 %s, 8589934591
  ret i64 %o
}

; CHECK-LABEL: ubfm_shift_mask:
; CHECK: kinsn_call bpf_arm64_ubfm_x
; CHECK-LABEL: ubfm_start_zero:
; CHECK: kinsn_call bpf_arm64_ubfm_x
; CHECK-LABEL: too_wide:
; CHECK-NOT: bpf_arm64_ubfm_x
; CHECK: exit
; CHECK-NOT: bpf_x86_
