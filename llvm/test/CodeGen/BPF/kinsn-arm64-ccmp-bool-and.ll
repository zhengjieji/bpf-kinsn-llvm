; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=all=disable \
; RUN:   -bpf-kinsn-mode=ccmp=force < %s | FileCheck %s

define dso_local i32 @and4_bool(i32 noundef %a, i32 noundef %b,
                                i32 noundef %c, i32 noundef %d)
                                local_unnamed_addr section "xdp" {
entry:
  %ca = icmp ne i32 %a, 0
  %cb = icmp ne i32 %b, 0
  %cc = icmp ne i32 %c, 0
  %cd = icmp ne i32 %d, 0
  %za = zext i1 %ca to i32
  %zb = zext i1 %cb to i32
  %zc = zext i1 %cc to i32
  %zd = zext i1 %cd to i32
  %ab = and i32 %za, %zb
  %abc = and i32 %ab, %zc
  %abcd = and i32 %abc, %zd
  ret i32 %abcd
}

define dso_local i32 @and_non_bool_deferred(i32 noundef %a, i32 noundef %b)
                                local_unnamed_addr section "xdp" {
entry:
  %and = and i32 %a, %b
  ret i32 %and
}

; CHECK-LABEL: and4_bool:
; CHECK: kinsn_call bpf_arm64_cmp_w
; CHECK-NEXT: kinsn_sidecar
; CHECK-NEXT: kinsn_call bpf_arm64_ccmp_w
; CHECK-NEXT: kinsn_sidecar
; CHECK-NEXT: kinsn_call bpf_arm64_ccmp_w
; CHECK-NEXT: kinsn_sidecar
; CHECK-NEXT: kinsn_call bpf_arm64_ccmp_w
; CHECK-NEXT: kinsn_sidecar
; CHECK-NEXT: kinsn_call bpf_arm64_cset_x_cond

; CHECK-LABEL: and_non_bool_deferred:
; CHECK-NOT: bpf_arm64_ccmp_w
; CHECK: exit
; CHECK-NOT: bpf_x86_
