# RustyArm64 Simulator

An ARMv8-A AArch64 functional simulator written in Rust.

## Goal

Run arbitrary bare-metal AArch64 ELF binaries correctly — including programs that use exception levels (EL0–EL3), the MMU, and system registers.

## Learning context

This project is primarily a vehicle for learning Rust. Unless explicitly asked to write code, the agent should explain concepts, point out issues, suggest approaches, and ask questions — not implement things. When Rust-specific guidance is relevant, teach the idiomatic way and explain why.

## Direction

- Implement the full AArch64 instruction set incrementally
- Add exception model: synchronous exceptions, IRQ/FIQ, ERET, SPSR/ELR per EL
- Add system registers (SCTLR, TCR, TTBR, etc.) and the MMU/page-table walker
- Add a UART or semihosting stub so programs can produce output
- Keep decode, execute, and CPU state cleanly separated as the ISA grows
