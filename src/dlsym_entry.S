#if defined(__x86_64__)

    .text
    .globl dlsym
    .type dlsym, @function
dlsym:
    .cfi_startproc
    endbr64
    cmpq $-1, %rdi
    jne .Ldispatch_x86_64

    /* Keep the original return address on the stack while resolving libc dlsym. */
    pushq %rdi
    .cfi_adjust_cfa_offset 8
    pushq %rsi
    .cfi_adjust_cfa_offset 8
    subq $8, %rsp
    .cfi_adjust_cfa_offset 8
    call libvgpu_get_real_dlsym
    addq $8, %rsp
    .cfi_adjust_cfa_offset -8
    popq %rsi
    .cfi_adjust_cfa_offset -8
    popq %rdi
    .cfi_adjust_cfa_offset -8

    testq %rax, %rax
    je .Lreturn_null_x86_64
    jmp *%rax

.Lreturn_null_x86_64:
    xorl %eax, %eax
    ret

.Ldispatch_x86_64:
    jmp libvgpu_dlsym_dispatch
    .cfi_endproc
    .size dlsym, .-dlsym

#elif defined(__aarch64__)

    .text
    .globl dlsym
    .type dlsym, %function
dlsym:
    .cfi_startproc
    .inst 0xd503245f  /* BTI c */
    cmn x0, #1
    b.ne .Ldispatch_aarch64

    /* Preserve the original arguments and link register across initialization. */
    stp x0, x1, [sp, #-32]!
    .cfi_adjust_cfa_offset 32
    str x30, [sp, #16]
    .cfi_offset x30, -16
    bl libvgpu_get_real_dlsym
    mov x16, x0
    ldr x30, [sp, #16]
    .cfi_restore x30
    ldp x0, x1, [sp], #32
    .cfi_adjust_cfa_offset -32

    cbz x16, .Lreturn_null_aarch64
    br x16

.Lreturn_null_aarch64:
    mov x0, xzr
    ret

.Ldispatch_aarch64:
    b libvgpu_dlsym_dispatch
    .cfi_endproc
    .size dlsym, .-dlsym

#endif

    .section .note.GNU-stack,"",%progbits
