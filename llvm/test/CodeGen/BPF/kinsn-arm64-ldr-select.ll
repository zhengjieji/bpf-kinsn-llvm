; RUN: llc -mtriple=bpfel -mcpu=v4 -verify-machineinstrs \
; RUN:   -bpf-enable-kinsn-select \
; RUN:   -bpf-kinsn-target=arm64 \
; RUN:   -bpf-kinsn-mode=all=disable \
; RUN:   -bpf-kinsn-mode=wide-load=force < %s | FileCheck %s

define dso_local i32 @load16_le(ptr nocapture noundef readonly %p) local_unnamed_addr section "xdp" {
entry:
  %p0 = getelementptr i8, ptr %p, i64 2
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i32
  %p1 = getelementptr i8, ptr %p, i64 3
  %b1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %b1 to i32
  %s1 = shl i32 %z1, 8
  %o1 = or i32 %z0, %s1
  %m = and i32 %o1, 65535
  ret i32 %m
}

; CHECK-LABEL: load16_le:
; CHECK: kinsn_sidecar 0, 33, 0
; CHECK: kinsn_call bpf_arm64_ldrh
; CHECK-NOT: bpf_x86_

define dso_local i32 @load32_le(ptr nocapture noundef readonly %p) local_unnamed_addr section "xdp" {
entry:
  %p0 = getelementptr i8, ptr %p, i64 4
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i32
  %p1 = getelementptr i8, ptr %p, i64 5
  %b1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %b1 to i32
  %s1 = shl i32 %z1, 8
  %o1 = or i32 %z0, %s1
  %p2 = getelementptr i8, ptr %p, i64 6
  %b2 = load i8, ptr %p2, align 1
  %z2 = zext i8 %b2 to i32
  %s2 = shl i32 %z2, 16
  %o2 = or i32 %o1, %s2
  %p3 = getelementptr i8, ptr %p, i64 7
  %b3 = load i8, ptr %p3, align 1
  %z3 = zext i8 %b3 to i32
  %s3 = shl i32 %z3, 24
  %o3 = or i32 %o2, %s3
  ret i32 %o3
}

; CHECK-LABEL: load32_le:
; CHECK: kinsn_sidecar 0, 65, 0
; CHECK: kinsn_call bpf_arm64_ldr_w
; CHECK-NOT: bpf_x86_

define dso_local i64 @load64_le(ptr nocapture noundef readonly %p) local_unnamed_addr section "xdp" {
entry:
  %p0 = getelementptr i8, ptr %p, i64 16
  %b0 = load i8, ptr %p0, align 1
  %z0 = zext i8 %b0 to i64
  %p1 = getelementptr i8, ptr %p, i64 17
  %b1 = load i8, ptr %p1, align 1
  %z1 = zext i8 %b1 to i64
  %s1 = shl i64 %z1, 8
  %o1 = or i64 %z0, %s1
  %p2 = getelementptr i8, ptr %p, i64 18
  %b2 = load i8, ptr %p2, align 1
  %z2 = zext i8 %b2 to i64
  %s2 = shl i64 %z2, 16
  %o2 = or i64 %o1, %s2
  %p3 = getelementptr i8, ptr %p, i64 19
  %b3 = load i8, ptr %p3, align 1
  %z3 = zext i8 %b3 to i64
  %s3 = shl i64 %z3, 24
  %o3 = or i64 %o2, %s3
  %p4 = getelementptr i8, ptr %p, i64 20
  %b4 = load i8, ptr %p4, align 1
  %z4 = zext i8 %b4 to i64
  %s4 = shl i64 %z4, 32
  %o4 = or i64 %o3, %s4
  %p5 = getelementptr i8, ptr %p, i64 21
  %b5 = load i8, ptr %p5, align 1
  %z5 = zext i8 %b5 to i64
  %s5 = shl i64 %z5, 40
  %o5 = or i64 %o4, %s5
  %p6 = getelementptr i8, ptr %p, i64 22
  %b6 = load i8, ptr %p6, align 1
  %z6 = zext i8 %b6 to i64
  %s6 = shl i64 %z6, 48
  %o6 = or i64 %o5, %s6
  %p7 = getelementptr i8, ptr %p, i64 23
  %b7 = load i8, ptr %p7, align 1
  %z7 = zext i8 %b7 to i64
  %s7 = shl i64 %z7, 56
  %o7 = or i64 %o6, %s7
  ret i64 %o7
}

; CHECK-LABEL: load64_le:
; CHECK: kinsn_sidecar 0, 257, 0
; CHECK: kinsn_call bpf_arm64_ldr_x
; CHECK-NOT: bpf_x86_
