from dataclasses import dataclass, field
from typing import Final

from unicorn import (
    UC_ARCH_X86, UC_ARCH_ARM, UC_ARCH_ARM64, UC_ARCH_MIPS, UC_ARCH_RISCV,
    UC_MODE_32, UC_MODE_64, UC_MODE_ARM, UC_MODE_MIPS32,
    UC_MODE_RISCV32, UC_MODE_RISCV64, UC_MODE_BIG_ENDIAN,
    UC_HOOK_INTR, UC_HOOK_INSN,
)
from unicorn.x86_const import (
    UC_X86_REG_EAX, UC_X86_REG_EBX, UC_X86_REG_ECX, UC_X86_REG_EDX,
    UC_X86_REG_ESI, UC_X86_REG_EDI, UC_X86_REG_EBP, UC_X86_REG_ESP,
    UC_X86_REG_EIP, UC_X86_REG_EFLAGS,
    UC_X86_REG_RAX, UC_X86_REG_RBX, UC_X86_REG_RCX, UC_X86_REG_RDX,
    UC_X86_REG_RSI, UC_X86_REG_RDI, UC_X86_REG_RBP, UC_X86_REG_RSP,
    UC_X86_REG_RIP,
    UC_X86_REG_R8, UC_X86_REG_R9, UC_X86_REG_R10, UC_X86_REG_R11,
    UC_X86_REG_R12, UC_X86_REG_R13, UC_X86_REG_R14, UC_X86_REG_R15,
    UC_X86_INS_SYSCALL,
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
from unicorn.mips_const import (
    UC_MIPS_REG_PC,
    UC_MIPS_REG_0, UC_MIPS_REG_1, UC_MIPS_REG_2, UC_MIPS_REG_3,
    UC_MIPS_REG_4, UC_MIPS_REG_5, UC_MIPS_REG_6, UC_MIPS_REG_7,
    UC_MIPS_REG_8, UC_MIPS_REG_9, UC_MIPS_REG_10, UC_MIPS_REG_11,
    UC_MIPS_REG_12, UC_MIPS_REG_13, UC_MIPS_REG_14, UC_MIPS_REG_15,
    UC_MIPS_REG_16, UC_MIPS_REG_17, UC_MIPS_REG_18, UC_MIPS_REG_19,
    UC_MIPS_REG_20, UC_MIPS_REG_21, UC_MIPS_REG_22, UC_MIPS_REG_23,
    UC_MIPS_REG_24, UC_MIPS_REG_25, UC_MIPS_REG_26, UC_MIPS_REG_27,
    UC_MIPS_REG_28, UC_MIPS_REG_29, UC_MIPS_REG_30, UC_MIPS_REG_31,
    UC_MIPS_REG_HI, UC_MIPS_REG_LO,
)
from unicorn.riscv_const import (
    UC_RISCV_REG_PC,
    UC_RISCV_REG_X0, UC_RISCV_REG_X1, UC_RISCV_REG_X2, UC_RISCV_REG_X3,
    UC_RISCV_REG_X4, UC_RISCV_REG_X5, UC_RISCV_REG_X6, UC_RISCV_REG_X7,
    UC_RISCV_REG_X8, UC_RISCV_REG_X9, UC_RISCV_REG_X10, UC_RISCV_REG_X11,
    UC_RISCV_REG_X12, UC_RISCV_REG_X13, UC_RISCV_REG_X14, UC_RISCV_REG_X15,
    UC_RISCV_REG_X16, UC_RISCV_REG_X17, UC_RISCV_REG_X18, UC_RISCV_REG_X19,
    UC_RISCV_REG_X20, UC_RISCV_REG_X21, UC_RISCV_REG_X22, UC_RISCV_REG_X23,
    UC_RISCV_REG_X24, UC_RISCV_REG_X25, UC_RISCV_REG_X26, UC_RISCV_REG_X27,
    UC_RISCV_REG_X28, UC_RISCV_REG_X29, UC_RISCV_REG_X30, UC_RISCV_REG_X31,
)
from capstone import (
    CS_ARCH_X86, CS_ARCH_ARM, CS_ARCH_ARM64, CS_ARCH_MIPS, CS_ARCH_RISCV,
    CS_MODE_32, CS_MODE_64, CS_MODE_ARM, CS_MODE_MIPS32,
    CS_MODE_RISCV32, CS_MODE_RISCV64, CS_MODE_BIG_ENDIAN,
)
from keystone import (
    KS_ARCH_X86, KS_ARCH_ARM, KS_ARCH_ARM64, KS_ARCH_MIPS,
    KS_MODE_32, KS_MODE_64, KS_MODE_ARM, KS_MODE_LITTLE_ENDIAN,
    KS_MODE_MIPS32, KS_MODE_BIG_ENDIAN,
)


@dataclass(frozen=True)
class ArchConfig:
    name: str
    uc_arch: int
    uc_mode: int
    cs_arch: int
    cs_mode: int
    register_map: dict[str, int] = field(default_factory=dict)
    pc_reg: str = ""
    sp_reg: str = ""
    endian: str = "little"  # "little" or "big"
    nop_bytes: bytes = b"\x90"  # architecture-specific NOP encoding
    # Keystone is optional — RISC-V has no Keystone backend.
    ks_arch: int | None = None
    ks_mode: int | None = None


# Shared MIPS32 register map (identical for LE and BE).
_MIPS32_REGISTER_MAP: Final[dict[str, int]] = {
    "zero": UC_MIPS_REG_0,
    "at": UC_MIPS_REG_1,
    "v0": UC_MIPS_REG_2,
    "v1": UC_MIPS_REG_3,
    "a0": UC_MIPS_REG_4,
    "a1": UC_MIPS_REG_5,
    "a2": UC_MIPS_REG_6,
    "a3": UC_MIPS_REG_7,
    "t0": UC_MIPS_REG_8,
    "t1": UC_MIPS_REG_9,
    "t2": UC_MIPS_REG_10,
    "t3": UC_MIPS_REG_11,
    "t4": UC_MIPS_REG_12,
    "t5": UC_MIPS_REG_13,
    "t6": UC_MIPS_REG_14,
    "t7": UC_MIPS_REG_15,
    "s0": UC_MIPS_REG_16,
    "s1": UC_MIPS_REG_17,
    "s2": UC_MIPS_REG_18,
    "s3": UC_MIPS_REG_19,
    "s4": UC_MIPS_REG_20,
    "s5": UC_MIPS_REG_21,
    "s6": UC_MIPS_REG_22,
    "s7": UC_MIPS_REG_23,
    "t8": UC_MIPS_REG_24,
    "t9": UC_MIPS_REG_25,
    "k0": UC_MIPS_REG_26,
    "k1": UC_MIPS_REG_27,
    "gp": UC_MIPS_REG_28,
    "sp": UC_MIPS_REG_29,
    "fp": UC_MIPS_REG_30,
    "ra": UC_MIPS_REG_31,
    "pc": UC_MIPS_REG_PC,
    "hi": UC_MIPS_REG_HI,
    "lo": UC_MIPS_REG_LO,
}

# Shared RISC-V register map (identical for RV32 and RV64).
_RISCV_REGISTER_MAP: Final[dict[str, int]] = {
    "zero": UC_RISCV_REG_X0,
    "ra": UC_RISCV_REG_X1,
    "sp": UC_RISCV_REG_X2,
    "gp": UC_RISCV_REG_X3,
    "tp": UC_RISCV_REG_X4,
    "t0": UC_RISCV_REG_X5,
    "t1": UC_RISCV_REG_X6,
    "t2": UC_RISCV_REG_X7,
    "s0": UC_RISCV_REG_X8,
    "s1": UC_RISCV_REG_X9,
    "a0": UC_RISCV_REG_X10,
    "a1": UC_RISCV_REG_X11,
    "a2": UC_RISCV_REG_X12,
    "a3": UC_RISCV_REG_X13,
    "a4": UC_RISCV_REG_X14,
    "a5": UC_RISCV_REG_X15,
    "a6": UC_RISCV_REG_X16,
    "a7": UC_RISCV_REG_X17,
    "s2": UC_RISCV_REG_X18,
    "s3": UC_RISCV_REG_X19,
    "s4": UC_RISCV_REG_X20,
    "s5": UC_RISCV_REG_X21,
    "s6": UC_RISCV_REG_X22,
    "s7": UC_RISCV_REG_X23,
    "s8": UC_RISCV_REG_X24,
    "s9": UC_RISCV_REG_X25,
    "s10": UC_RISCV_REG_X26,
    "s11": UC_RISCV_REG_X27,
    "t3": UC_RISCV_REG_X28,
    "t4": UC_RISCV_REG_X29,
    "t5": UC_RISCV_REG_X30,
    "t6": UC_RISCV_REG_X31,
    "pc": UC_RISCV_REG_PC,
}


ARCHITECTURES: Final[dict[str, ArchConfig]] = {
    "x86_32": ArchConfig(
        name="x86_32",
        uc_arch=UC_ARCH_X86,
        uc_mode=UC_MODE_32,
        cs_arch=CS_ARCH_X86,
        cs_mode=CS_MODE_32,
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
        ks_arch=KS_ARCH_X86,
        ks_mode=KS_MODE_32,
    ),
    "x86_64": ArchConfig(
        name="x86_64",
        uc_arch=UC_ARCH_X86,
        uc_mode=UC_MODE_64,
        cs_arch=CS_ARCH_X86,
        cs_mode=CS_MODE_64,
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
        ks_arch=KS_ARCH_X86,
        ks_mode=KS_MODE_64,
    ),
    "arm": ArchConfig(
        name="arm",
        uc_arch=UC_ARCH_ARM,
        uc_mode=UC_MODE_ARM,
        cs_arch=CS_ARCH_ARM,
        cs_mode=CS_MODE_ARM,
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
        ks_arch=KS_ARCH_ARM,
        ks_mode=KS_MODE_ARM,
        nop_bytes=b"\x00\xf0\x20\xe3",
    ),
    "arm64": ArchConfig(
        name="arm64",
        uc_arch=UC_ARCH_ARM64,
        uc_mode=UC_MODE_ARM,
        cs_arch=CS_ARCH_ARM64,
        cs_mode=CS_MODE_ARM,
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
        ks_arch=KS_ARCH_ARM64,
        ks_mode=KS_MODE_LITTLE_ENDIAN,
        nop_bytes=b"\x1f\x20\x03\xd5",
    ),
    "mips32": ArchConfig(
        name="mips32",
        uc_arch=UC_ARCH_MIPS,
        uc_mode=UC_MODE_MIPS32,
        cs_arch=CS_ARCH_MIPS,
        cs_mode=CS_MODE_MIPS32,
        register_map=_MIPS32_REGISTER_MAP,
        pc_reg="pc",
        sp_reg="sp",
        ks_arch=KS_ARCH_MIPS,
        ks_mode=KS_MODE_MIPS32,
        nop_bytes=b"\x00\x00\x00\x00",
    ),
    "mips32be": ArchConfig(
        name="mips32be",
        uc_arch=UC_ARCH_MIPS,
        uc_mode=UC_MODE_MIPS32 | UC_MODE_BIG_ENDIAN,
        cs_arch=CS_ARCH_MIPS,
        cs_mode=CS_MODE_MIPS32 | CS_MODE_BIG_ENDIAN,
        register_map=_MIPS32_REGISTER_MAP,
        pc_reg="pc",
        sp_reg="sp",
        ks_arch=KS_ARCH_MIPS,
        ks_mode=KS_MODE_MIPS32 | KS_MODE_BIG_ENDIAN,
        endian="big",
        nop_bytes=b"\x00\x00\x00\x00",
    ),
    "riscv32": ArchConfig(
        name="riscv32",
        uc_arch=UC_ARCH_RISCV,
        uc_mode=UC_MODE_RISCV32,
        cs_arch=CS_ARCH_RISCV,
        cs_mode=CS_MODE_RISCV32,
        register_map=_RISCV_REGISTER_MAP,
        pc_reg="pc",
        sp_reg="sp",
        # No Keystone backend for RISC-V.
        nop_bytes=b"\x13\x00\x00\x00",
    ),
    "riscv64": ArchConfig(
        name="riscv64",
        uc_arch=UC_ARCH_RISCV,
        uc_mode=UC_MODE_RISCV64,
        cs_arch=CS_ARCH_RISCV,
        cs_mode=CS_MODE_RISCV64,
        register_map=_RISCV_REGISTER_MAP,
        pc_reg="pc",
        sp_reg="sp",
        # No Keystone backend for RISC-V.
        nop_bytes=b"\x13\x00\x00\x00",
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


# -- Syscall conventions per architecture ------------------------------------

@dataclass(frozen=True)
class SyscallConvention:
    """Architecture-specific syscall/interrupt convention."""
    hook_type: int          # UC_HOOK_INTR or UC_HOOK_INSN
    hook_arg: int | None    # aux1 for UC_HOOK_INSN (e.g., UC_X86_INS_SYSCALL)
    intno_filter: int | None  # filter INTR by interrupt number (None = match all)
    nr_reg: str             # register holding syscall number
    arg_regs: tuple[str, ...]  # registers for syscall arguments
    ret_reg: str            # register for return value


SYSCALL_CONVENTIONS: Final[dict[str, SyscallConvention]] = {
    "x86_32":   SyscallConvention(UC_HOOK_INTR, None, 0x80, "eax", ("ebx", "ecx", "edx", "esi", "edi", "ebp"), "eax"),
    "x86_64":   SyscallConvention(UC_HOOK_INSN, UC_X86_INS_SYSCALL, None, "rax", ("rdi", "rsi", "rdx", "r10", "r8", "r9"), "rax"),
    "arm":      SyscallConvention(UC_HOOK_INTR, None, 2, "r7", ("r0", "r1", "r2", "r3", "r4", "r5"), "r0"),
    "arm64":    SyscallConvention(UC_HOOK_INTR, None, 2, "x8", ("x0", "x1", "x2", "x3", "x4", "x5"), "x0"),
    "mips32":   SyscallConvention(UC_HOOK_INTR, None, 17, "v0", ("a0", "a1", "a2", "a3"), "v0"),
    "mips32be": SyscallConvention(UC_HOOK_INTR, None, 17, "v0", ("a0", "a1", "a2", "a3"), "v0"),
    "riscv32":  SyscallConvention(UC_HOOK_INTR, None, 8, "a7", ("a0", "a1", "a2", "a3", "a4", "a5"), "a0"),
    "riscv64":  SyscallConvention(UC_HOOK_INTR, None, 8, "a7", ("a0", "a1", "a2", "a3", "a4", "a5"), "a0"),
}
