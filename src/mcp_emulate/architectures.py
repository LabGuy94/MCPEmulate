from dataclasses import dataclass
from typing import Final

from unicorn import UC_ARCH_X86, UC_ARCH_ARM, UC_ARCH_ARM64, UC_MODE_32, UC_MODE_64, UC_MODE_ARM
from unicorn.x86_const import (
    UC_X86_REG_EAX, UC_X86_REG_EBX, UC_X86_REG_ECX, UC_X86_REG_EDX,
    UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_EBP, UC_X86_REG_ESP,
    UC_X86_REG_EIP, UC_X86_REG_EFLAGS,
    UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RCX, UC_X86_REG_RDX,
    UC_X86_REG_RSI, UC_X86_REG_RDI, UC_X86_REG_RBP, UC_X86_REG_RSP,
    UC_X86_REG_RIP,
    UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R11,
    UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
)
from unicorn.arm_const import (
    UC_ARM_REG_R0, UC_ARM_REG_R1, UC_ARM_REG_R2, UC_ARM_REG_R3,
    UC_ARM_REG_R4, UC_ARM_REG_R5, UC_ARM_REG_R6, UC_ARM_REG_R7,
    UC_ARM_REG_R8, UC_ARM_REG_R9, UC_ARM_REG_R10, UC_ARM_REG_R11,
    UC_ARM_REG_R12, UC_ARM_REG_SP, UC_ARM_REG_LR, UC_ARM_REG_PC,
    UC_ARM_REG_CPSR,
)
from unicorn.arm64_const import (
    UC_ARM64_REG_X0, UC_ARM64_REG_X1, UC_ARM64_REG_X2, UC_ARM64_REG_X3,
    UC_ARM64_REG_X4, UC_ARM64_REG_X5, UC_ARM64_REG_X6, UC_ARM64_REG_X7,
    UC_ARM64_REG_X8, UC_ARM64_REG_X9, UC_ARM64_REG_X10, UC_ARM64_REG_X11,
    UC_ARM64_REG_X12, UC_ARM64_REG_X13, UC_ARM64_REG_X14, UC_ARM64_REG_X15,
    UC_ARM64_REG_X16, UC_ARM64_REG_X17, UC_ARM64_REG_X18, UC_ARM64_REG_X19,
    UC_ARM64_REG_X20, UC_ARM64_REG_X21, UC_ARM64_REG_X22, UC_ARM64_REG_X23,
    UC_ARM64_REG_X24, UC_ARM64_REG_X25, UC_ARM64_REG_X26, UC_ARM64_REG_X27,
    UC_ARM64_REG_X28, UC_ARM64_REG_X29, UC_ARM64_REG_X30,
    UC_ARM64_REG_SP, UC_ARM64_REG_PC, UC_ARM64_REG_NZCV,
)
from capstone import CS_ARCH_X86, CS_ARCH_ARM, CS_ARCH_ARM64, CS_MODE_32, CS_MODE_64, CS_MODE_ARM
from keystone import KS_ARCH_X86, KS_ARCH_ARM, KS_ARCH_ARM64, KS_MODE_32, KS_MODE_64, KS_MODE_ARM, KS_MODE_LITTLE_ENDIAN


@dataclass(frozen=True)
class ArchConfig:
    name: str
    uc_arch: int
    uc_mode: int
    cs_arch: int
    cs_mode: int
    ks_arch: int
    ks_mode: int
    register_map: dict[str, int]
    pc_reg: str
    sp_reg: str


ARCHITECTURES: Final[dict[str, ArchConfig]] = {
    "x86_32": ArchConfig(
        name="x86_32",
        uc_arch=UC_ARCH_X86,
        uc_mode=UC_MODE_32,
        cs_arch=CS_ARCH_X86,
        cs_mode=CS_MODE_32,
        ks_arch=KS_ARCH_X86,
        ks_mode=KS_MODE_32,
        register_map={
            "eax": UC_X86_REG_EAX,
            "ebx": UC_X86_REG_EBX,
            "ecx": UC_X86_REG_ECX,
            "edx": UC_X86_REG_EDX,
            "esi": UC_X86_REG_ESI,
            "edi": UC_X86_REG_EDI,
            "ebp": UC_X86_REG_EBP,
            "esp": UC_X86_REG_ESP,
            "eip": UC_X86_REG_EIP,
            "eflags": UC_X86_REG_EFLAGS,
        },
        pc_reg="eip",
        sp_reg="esp",
    ),
    "x86_64": ArchConfig(
        name="x86_64",
        uc_arch=UC_ARCH_X86,
        uc_mode=UC_MODE_64,
        cs_arch=CS_ARCH_X86,
        cs_mode=CS_MODE_64,
        ks_arch=KS_ARCH_X86,
        ks_mode=KS_MODE_64,
        register_map={
            "rax": UC_X86_REG_RAX,
            "rbx": UC_X86_REG_RBX,
            "rcx": UC_X86_REG_RCX,
            "rdx": UC_X86_REG_RDX,
            "rsi": UC_X86_REG_RSI,
            "rdi": UC_X86_REG_RDI,
            "rbp": UC_X86_REG_RBP,
            "rsp": UC_X86_REG_RSP,
            "rip": UC_X86_REG_RIP,
            "r8":  UC_X86_REG_R8,
            "r9":  UC_X86_REG_R9,
            "r10": UC_X86_REG_R10,
            "r11": UC_X86_REG_R11,
            "r12": UC_X86_REG_R12,
            "r13": UC_X86_REG_R13,
            "r14": UC_X86_REG_R14,
            "r15": UC_X86_REG_R15,
            "eflags": UC_X86_REG_EFLAGS,
        },
        pc_reg="rip",
        sp_reg="rsp",
    ),
    "arm": ArchConfig(
        name="arm",
        uc_arch=UC_ARCH_ARM,
        uc_mode=UC_MODE_ARM,
        cs_arch=CS_ARCH_ARM,
        cs_mode=CS_MODE_ARM,
        ks_arch=KS_ARCH_ARM,
        ks_mode=KS_MODE_ARM,
        register_map={
            "r0":  UC_ARM_REG_R0,
            "r1":  UC_ARM_REG_R1,
            "r2":  UC_ARM_REG_R2,
            "r3":  UC_ARM_REG_R3,
            "r4":  UC_ARM_REG_R4,
            "r5":  UC_ARM_REG_R5,
            "r6":  UC_ARM_REG_R6,
            "r7":  UC_ARM_REG_R7,
            "r8":  UC_ARM_REG_R8,
            "r9":  UC_ARM_REG_R9,
            "r10": UC_ARM_REG_R10,
            "r11": UC_ARM_REG_R11,
            "r12": UC_ARM_REG_R12,
            "sp":  UC_ARM_REG_SP,
            "lr":  UC_ARM_REG_LR,
            "pc":  UC_ARM_REG_PC,
            "cpsr": UC_ARM_REG_CPSR,
        },
        pc_reg="pc",
        sp_reg="sp",
    ),
    "arm64": ArchConfig(
        name="arm64",
        uc_arch=UC_ARCH_ARM64,
        uc_mode=UC_MODE_ARM,
        cs_arch=CS_ARCH_ARM64,
        cs_mode=CS_MODE_ARM,
        ks_arch=KS_ARCH_ARM64,
        ks_mode=KS_MODE_LITTLE_ENDIAN,
        register_map={
            "x0":  UC_ARM64_REG_X0,
            "x1":  UC_ARM64_REG_X1,
            "x2":  UC_ARM64_REG_X2,
            "x3":  UC_ARM64_REG_X3,
            "x4":  UC_ARM64_REG_X4,
            "x5":  UC_ARM64_REG_X5,
            "x6":  UC_ARM64_REG_X6,
            "x7":  UC_ARM64_REG_X7,
            "x8":  UC_ARM64_REG_X8,
            "x9":  UC_ARM64_REG_X9,
            "x10": UC_ARM64_REG_X10,
            "x11": UC_ARM64_REG_X11,
            "x12": UC_ARM64_REG_X12,
            "x13": UC_ARM64_REG_X13,
            "x14": UC_ARM64_REG_X14,
            "x15": UC_ARM64_REG_X15,
            "x16": UC_ARM64_REG_X16,
            "x17": UC_ARM64_REG_X17,
            "x18": UC_ARM64_REG_X18,
            "x19": UC_ARM64_REG_X19,
            "x20": UC_ARM64_REG_X20,
            "x21": UC_ARM64_REG_X21,
            "x22": UC_ARM64_REG_X22,
            "x23": UC_ARM64_REG_X23,
            "x24": UC_ARM64_REG_X24,
            "x25": UC_ARM64_REG_X25,
            "x26": UC_ARM64_REG_X26,
            "x27": UC_ARM64_REG_X27,
            "x28": UC_ARM64_REG_X28,
            "x29": UC_ARM64_REG_X29,
            "x30": UC_ARM64_REG_X30,
            "sp":  UC_ARM64_REG_SP,
            "pc":  UC_ARM64_REG_PC,
            "cpsr": UC_ARM64_REG_NZCV,
        },
        pc_reg="pc",
        sp_reg="sp",
    ),
}


def get_arch(name: str) -> ArchConfig:
    """Look up an architecture config by name.

    Raises ValueError if name is not a known architecture.
    """
    try:
        return ARCHITECTURES[name]
    except KeyError:
        valid = ", ".join(sorted(ARCHITECTURES))
        raise ValueError(f"Unknown architecture {name!r}. Valid options: {valid}") from None
