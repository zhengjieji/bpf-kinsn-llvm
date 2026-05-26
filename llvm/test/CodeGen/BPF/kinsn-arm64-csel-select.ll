; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=all=disable \
; RUN:   -bpf-kinsn-mode=cmov=force < %s | FileCheck %s

define dso_local i64 @select_ne_zero(i64 noundef %cond, i64 noundef %a, i64 noundef %b) local_unnamed_addr section "xdp" {
entry:
  %cmp = icmp ne i64 %cond, 0
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define dso_local i64 @select_eq_zero(i64 noundef %cond, i64 noundef %a, i64 noundef %b) local_unnamed_addr section "xdp" {
entry:
  %cmp = icmp eq i64 %cond, 0
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define dso_local i64 @select_ne_nonzero_deferred(i64 noundef %cond, i64 noundef %a, i64 noundef %b) local_unnamed_addr section "xdp" {
entry:
  %cmp = icmp ne i64 %cond, 7
  %sel = select i1 %cmp, i64 %a, i64 %b
  ret i64 %sel
}

define dso_local i32 @select_i32_value_deferred(i64 noundef %cond, i32 noundef %a, i32 noundef %b) local_unnamed_addr section "xdp" {
entry:
  %cmp = icmp ne i64 %cond, 0
  %sel = select i1 %cmp, i32 %a, i32 %b
  ret i32 %sel
}

; CHECK-LABEL: select_ne_zero:
; CHECK: kinsn_sidecar 1, 0, 0
; CHECK-NEXT: kinsn_call bpf_arm64_tst
; CHECK-NEXT: kinsn_sidecar 0, 306, 0
; CHECK-NEXT: kinsn_call bpf_arm64_csel_ne

; CHECK-LABEL: select_eq_zero:
; CHECK: kinsn_sidecar 1, 0, 0
; CHECK-NEXT: kinsn_call bpf_arm64_tst
; CHECK-NEXT: kinsn_sidecar 0, 291, 0
; CHECK-NEXT: kinsn_call bpf_arm64_csel_ne

; CHECK-LABEL: select_ne_nonzero_deferred:
; CHECK-NOT: bpf_arm64_csel_ne
; CHECK: exit

; CHECK-LABEL: select_i32_value_deferred:
; CHECK-NOT: bpf_arm64_csel_ne
; CHECK: exit
; CHECK-NOT: bpf_x86_
