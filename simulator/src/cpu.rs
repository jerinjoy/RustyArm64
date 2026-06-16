use crate::decode::{self, DecodeError, Instruction};
use crate::executor::{execute_instruction, ExecError};
use crate::memory::Memory;
use crate::registers::Registers;

/// Errors that can occur during CPU execution.
#[derive(Debug, PartialEq, Eq)]
pub enum CpuError {
    /// The program executed a HLT instruction.
    Halt,
    /// An unrecognized instruction word was encountered.
    UnknownInstruction(u32),
    /// Memory access at the given address was out of bounds.
    MemoryFault(u64),
}

impl std::fmt::Display for CpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Halt => write!(f, "CPU halted"),
            Self::UnknownInstruction(w) => write!(f, "unknown instruction: 0x{w:08X}"),
            Self::MemoryFault(a) => write!(f, "memory fault at address 0x{a:016X}"),
        }
    }
}

impl std::error::Error for CpuError {}

/// Represents the state of an ARMv8-A AArch64 CPU.
pub struct Cpu {
    /// Architectural register file (X0–X30, SP, PC, flags).
    pub regs: Registers,
    /// Physical memory.
    pub mem: Memory,
    /// True when the processor has halted (HLT executed).
    pub halted: bool,
}

impl Cpu {
    /// Create a new CPU with the given memory, all registers zeroed,
    /// PC = 0, and `halted = false`.
    pub fn new(memory: Memory) -> Self {
        Self {
            regs: Registers::new(),
            mem: memory,
            halted: false,
        }
    }

    // ── Convenience accessors (delegate to regs) ─────────────────────

    /// Read a general-purpose register (X0-X30).
    ///
    /// # Panics
    /// Panics if `index` is greater than 30.
    pub fn read_reg(&self, index: usize) -> u64 {
        assert!(index <= 30, "register index {} out of range (0..=30)", index);
        self.regs.read(index as u8)
    }

    /// Write a general-purpose register (X0-X30).
    ///
    /// # Panics
    /// Panics if `index` is greater than 30.
    pub fn write_reg(&mut self, index: usize, value: u64) {
        assert!(index <= 30, "register index {} out of range (0..=30)", index);
        self.regs.write(index as u8, value);
    }

    /// Read the stack pointer (SP).
    pub fn read_sp(&self) -> u64 {
        self.regs.read_sp()
    }

    /// Write the stack pointer (SP).
    pub fn write_sp(&mut self, value: u64) {
        self.regs.write_sp(value);
    }

    /// Read the program counter.
    pub fn read_pc(&self) -> u64 {
        self.regs.read_pc()
    }

    /// Write the program counter.
    pub fn write_pc(&mut self, value: u64) {
        self.regs.write_pc(value);
    }

    /// Read PSTATE as a packed `u64`.
    ///
    /// NZCV flags occupy bits [31:28]; all other bits are zero.
    pub fn read_pstate(&self) -> u64 {
        let n = if self.regs.n { 1u64 << 31 } else { 0 };
        let z = if self.regs.z { 1u64 << 30 } else { 0 };
        let c = if self.regs.c { 1u64 << 29 } else { 0 };
        let v = if self.regs.v { 1u64 << 28 } else { 0 };
        n | z | c | v
    }

    /// Write PSTATE from a packed `u64`.
    ///
    /// Only bits [31:28] are examined; other bits are ignored.
    pub fn write_pstate(&mut self, value: u64) {
        self.regs.n = (value >> 31) & 1 == 1;
        self.regs.z = (value >> 30) & 1 == 1;
        self.regs.c = (value >> 29) & 1 == 1;
        self.regs.v = (value >> 28) & 1 == 1;
    }

    // ── Instruction execution ────────────────────────────────────────

    /// Fetch a 32-bit instruction from memory at the current PC.
    ///
    /// Reads 4 bytes in little-endian order. Does not advance PC.
    /// Returns `Err(CpuError::MemoryFault)` if the read goes out of bounds.
    pub fn fetch(&self, mem: &[u8]) -> Result<u32, CpuError> {
        let pc = self.read_pc() as usize;
        if pc + 4 > mem.len() {
            return Err(CpuError::MemoryFault(self.read_pc()));
        }
        let b0 = mem[pc] as u32;
        let b1 = mem[pc + 1] as u32;
        let b2 = mem[pc + 2] as u32;
        let b3 = mem[pc + 3] as u32;
        Ok(b0 | (b1 << 8) | (b2 << 16) | (b3 << 24))
    }

    /// Execute a decoded instruction, reading/writing registers and memory.
    ///
    /// Returns `Ok(())` on success, or `Err(CpuError::Halt)` for HLT.
    /// Unknown instructions are handled before calling this method (see
    /// [`step`](Self::step)).
    pub fn execute(
        &mut self,
        _mem: &mut [u8],
        inst: Instruction,
    ) -> Result<(), CpuError> {
        match execute_instruction(&mut self.regs, &mut self.mem, &mut self.halted, inst) {
            Ok(()) => {
                if self.halted {
                    Err(CpuError::Halt)
                } else {
                    Ok(())
                }
            }
            Err(ExecError::MemoryFault(addr)) => Err(CpuError::MemoryFault(addr)),
        }
    }

    /// Fetch, decode, and execute a single instruction (a "step").
    ///
    /// On success, advances PC by 4.
    /// On `HLT`, returns `Err(CpuError::Halt)` without modifying PC.
    /// On a decode failure, returns
    /// `Err(CpuError::UnknownInstruction)` without modifying PC.
    /// On a memory fault during fetch, returns
    /// `Err(CpuError::MemoryFault)` without modifying PC.
    pub fn step(&mut self, mem: &mut [u8]) -> Result<(), CpuError> {
        let inst_word = self.fetch(mem)?;

        // Decode the instruction word.
        let inst = match decode::decode(inst_word) {
            Ok(inst) => inst,
            Err(DecodeError::UnknownOpcode(_) | DecodeError::IllegalEncoding) => {
                return Err(CpuError::UnknownInstruction(inst_word));
            }
        };

        match self.execute(mem, inst) {
            Err(CpuError::Halt) => Err(CpuError::Halt),
            Ok(()) => {
                self.write_pc(self.read_pc().wrapping_add(4));
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    // ── Main run loop ────────────────────────────────────────────────

    /// Run the CPU until it halts or encounters an error.
    ///
    /// The execution loop:
    /// 1. Fetches a 32‑bit word from `self.mem` at `self.regs.pc`.
    /// 2. Decodes it.
    /// 3. Increments `self.regs.pc` by 4.
    /// 4. Executes the instruction.
    /// 5. If `halted` is true, breaks the loop.
    ///
    /// Returns `Ok(())` on a clean halt, or `Err(CpuError)` on error.
    pub fn run(&mut self) -> Result<(), CpuError> {
        loop {
            // Fetch from internal memory.
            let pc = self.read_pc();
            let inst_word = self
                .mem
                .read_u32(pc)
                .map_err(|_| CpuError::MemoryFault(pc))?;

            // Decode.
            let inst = match decode::decode(inst_word) {
                Ok(inst) => inst,
                Err(DecodeError::UnknownOpcode(_) | DecodeError::IllegalEncoding) => {
                    return Err(CpuError::UnknownInstruction(inst_word));
                }
            };

            // Increment PC before execution.
            self.write_pc(pc.wrapping_add(4));

            // Execute.
            execute_instruction(&mut self.regs, &mut self.mem, &mut self.halted, inst)
                .map_err(|e| match e {
                    ExecError::MemoryFault(addr) => CpuError::MemoryFault(addr),
                })?;

            if self.halted {
                break;
            }
        }
        Ok(())
    }

    /// Print the complete register state to stdout.
    ///
    /// Outputs X0–X30, SP, PC, and the N/Z/C/V flags.
    pub fn print_registers(&self) {
        println!("{}", self.regs.dump());
    }
}

impl Default for Cpu {
    fn default() -> Self {
        Self::new(Memory::new(0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: store a 32-bit word into a &mut [u8] at `addr` ───────

    fn write_inst(mem: &mut [u8], addr: u64, word: u32) {
        let a = addr as usize;
        mem[a] = word as u8;
        mem[a + 1] = (word >> 8) as u8;
        mem[a + 2] = (word >> 16) as u8;
        mem[a + 3] = (word >> 24) as u8;
    }

    // ── Existing CPU tests ───────────────────────────────────────────

    #[test]
    fn test_new_cpu_all_zero() {
        let cpu = Cpu::new(Memory::new(64));
        for i in 0..=30 {
            assert_eq!(cpu.read_reg(i), 0, "register X{} should be 0", i);
        }
        assert_eq!(cpu.read_sp(), 0);
        assert_eq!(cpu.read_pc(), 0);
        assert_eq!(cpu.read_pstate(), 0);
    }

    #[test]
    fn test_write_and_read_reg() {
        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_reg(0, 0xDEAD_BEEF);
        assert_eq!(cpu.read_reg(0), 0xDEAD_BEEF);

        cpu.write_reg(30, 0xCAFE);
        assert_eq!(cpu.read_reg(30), 0xCAFE);
    }

    #[test]
    fn test_write_and_read_sp() {
        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_sp(0xFFFF_0000);
        assert_eq!(cpu.read_sp(), 0xFFFF_0000);
    }

    #[test]
    fn test_write_and_read_pc() {
        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_pc(0x8000);
        assert_eq!(cpu.read_pc(), 0x8000);
    }

    #[test]
    fn test_write_and_read_pstate() {
        let mut cpu = Cpu::new(Memory::new(64));
        // Set N=1, Z=1, C=0, V=0 via packed PSTATE.
        cpu.write_pstate(0xA000_0000); // N=1, Z=1
        assert_eq!(cpu.read_pstate(), 0xA000_0000);
    }

    #[test]
    fn test_default_trait() {
        let cpu: Cpu = Default::default();
        assert_eq!(cpu.read_pc(), 0);
    }

    #[test]
    #[should_panic(expected = "register index 31 out of range")]
    fn test_read_reg_out_of_range() {
        let cpu = Cpu::new(Memory::new(64));
        cpu.read_reg(31);
    }

    #[test]
    #[should_panic(expected = "register index 31 out of range")]
    fn test_write_reg_out_of_range() {
        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_reg(31, 0);
    }

    // ── Step tests ────────────────────────────────────────────────────

    #[test]
    fn test_step_add() {
        // ADD X0, X1, #5  →  0x91001420
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x9100_1420);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_reg(1, 10); // X1 = 10
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert!(result.is_ok(), "step should succeed");

        // X0 = 10 + 5 = 15
        assert_eq!(cpu.read_reg(0), 15);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_step_sub() {
        // SUB X2, X3, #4095, LSL #12  →  0xD17FFC62
        // 4095 << 12 = 0xFFF000
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0xD17F_FC62);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_reg(3, 0x100_0000); // X3 = 0x1000000
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert!(result.is_ok(), "step should succeed");

        // X2 = 0x1000000 - 0xFFF000 = 0x1000
        assert_eq!(cpu.read_reg(2), 0x1000);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_step_movz() {
        // MOVZ X0, #0x42, LSL #0  →  0xD2800840
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0xD280_0840);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert!(result.is_ok(), "step should succeed");

        assert_eq!(cpu.read_reg(0), 0x42);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_step_hlt() {
        // HLT #0  →  0xD4030000
        // Memory must be large enough to cover PC (0x100) + 4.
        let mut mem = vec![0u8; 0x104];
        write_inst(&mut mem, 0x100, 0xD403_0000);

        let mut cpu = Cpu::new(Memory::new(0x104));
        cpu.write_pc(0x100);

        let result = cpu.step(&mut mem);
        assert_eq!(result, Err(CpuError::Halt));
        // PC must not change on HLT.
        assert_eq!(cpu.read_pc(), 0x100);
    }

    #[test]
    fn test_step_multiple() {
        // Sequence:
        //   0x00: MOVZ X5, #0x42  →  0xD2800845  (write 0x42 into X5)
        //   0x04: ADD X5, X5, #1  →  0x910004A5  (X5 = X5 + 1 → 0x43)
        //   0x08: HLT #0          →  0xD4030000

        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0x00, 0xD280_0845); // MOVZ X5, #0x42
        write_inst(&mut mem, 0x04, 0x9100_04A5); // ADD X5, X5, #1
        write_inst(&mut mem, 0x08, 0xD403_0000); // HLT

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_pc(0x00);

        // Step 1: MOVZ
        cpu.step(&mut mem).expect("step 1 (MOVZ)");
        assert_eq!(cpu.read_reg(5), 0x42);
        assert_eq!(cpu.read_pc(), 0x04);

        // Step 2: ADD
        cpu.step(&mut mem).expect("step 2 (ADD)");
        assert_eq!(cpu.read_reg(5), 0x43);
        assert_eq!(cpu.read_pc(), 0x08);

        // Step 3: HLT
        let result = cpu.step(&mut mem);
        assert_eq!(result, Err(CpuError::Halt));
        assert_eq!(cpu.read_pc(), 0x08);
    }

    #[test]
    fn test_fetch_out_of_bounds() {
        // Memory has only 4 bytes; PC = 8 is out of bounds.
        let mem = vec![0u8; 4];
        let mut cpu = Cpu::new(Memory::new(4));
        cpu.write_pc(8);
        let result = cpu.fetch(&mem);
        assert!(matches!(result, Err(CpuError::MemoryFault(8))));
    }

    #[test]
    fn test_step_unknown_instruction() {
        // 0x00000000 is not a valid decoded instruction → UnknownOpcode
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x0000_0000);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_pc(0);

        let result = cpu.step(&mut mem);
        assert_eq!(result, Err(CpuError::UnknownInstruction(0x0000_0000)));
        // PC must not advance.
        assert_eq!(cpu.read_pc(), 0);
    }

    // ── Register-31 semantics (architecturally correct: XZR) ────────

    #[test]
    fn test_execute_add_sub_rd_31_discards_write() {
        // ADD XZR, X1, #5  →  encoding with rd=31
        // sf=1, op=0, S=0, sh=0, imm12=5, rn=1, rd=31
        // = 0x9100143F
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x9100_143F);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_reg(1, 100);
        cpu.write_sp(0xFFFF);
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        // rd=31 → XZR, write discarded. SP unchanged.
        assert_eq!(cpu.read_sp(), 0xFFFF);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_execute_add_sub_rn_31_reads_zero() {
        // ADD X0, XZR, #16  →  rn=31
        // sf=1, op=0, sh=0, imm12=16 (0x010), rn=31, rd=0
        // = 0x9100401F → wait that has rd=31 too. Let me compute:
        // sf=1, op=0, S=0, sh=0, imm12=16 (0b000000010000), rn=31 (11111), rd=0 (00000)
        // Word: 1 0 0 10001 00 000000010000 11111 00000
        // = 0x9100401F ? No let me recalculate.
        // bits[31]=1, [30]=0, [29]=0, [28:24]=10001, [23:22]=00, [21:10]=000000010000, [9:5]=11111, [4:0]=00000
        // 31: 1
        // 30: 0
        // 29: 0
        // 28: 1
        // 27: 0
        // 26: 0
        // 25: 0
        // 24: 1
        // 23: 0
        // 22: 0
        // 21: 0
        // 20: 0
        // 19: 0
        // 18: 0
        // 17: 0
        // 16: 0
        // 15: 0
        // 14: 0
        // 13: 0
        // 12: 1
        // 11: 0
        // 10: 0
        // 9: 1
        // 8: 1
        // 7: 1
        // 6: 1
        // 5: 1
        // 4: 0
        // 3: 0
        // 2: 0
        // 1: 0
        // 0: 0
        // = 0x9100_03E0
        // Let me verify: 1001_0001_0000_0000_0000_0011_1110_0000 = 0x910003E0
        // Actually imm12=16 is bit[21:10] = 0x010, so bits 21:10 = 0000_0001_0000
        // Let me recalculate: sf=1, op=0, S=0 → bits[31:29] = 100
        // [28:24] = 10001
        // [23:22] = 00
        // [21:10] = 00_0001_0000 (imm12=16 shifted left by 0? No, imm12 occupies bits[21:10])
        // Wait, bit 10 is LSB of imm12. imm12=16 = 0b0000_0001_0000
        // So bits[21:10] = 0000_0001_0000
        // [9:5] = 11111 (rn=31)
        // [4:0] = 00000 (rd=0)
        // 100_10001_00_0000010000_11111_00000
        // = 0x9100_43E0 ? Let me just compute using the old test value.
        // Actually the old test used 0x910043E1 for ADD X1, SP, #16 (rn=31, rd=1)
        // For rd=0: the lower 5 bits change from 00001 to 00000.
        // So 0x910043E1 → change rd from 1 to 0 → 0x910043E0
        // Let me just use that.
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x9100_43E0);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_sp(0xFFFF); // SP has a value but won't be read
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        // rn=31 → XZR reads 0. 0 + 16 = 16.
        assert_eq!(cpu.read_reg(0), 16);
        assert_eq!(cpu.read_pc(), 4);
    }

    #[test]
    fn test_execute_movz_rd_31_discards_write() {
        // MOVZ XZR, #0x42  →  rd=31
        // = 0xD280085F
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0xD280_085F);

        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_sp(0xFFFF);
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        // rd=31 → XZR, write discarded. SP unchanged.
        assert_eq!(cpu.read_sp(), 0xFFFF);
    }

    // ── 32-bit tests ─────────────────────────────────────────────────

    #[test]
    fn test_execute_add_32bit() {
        // ADD W0, W1, #5  (32-bit) → 0x11001420
        let mut mem = vec![0u8; 64];
        write_inst(&mut mem, 0, 0x1100_1420);

        let mut cpu = Cpu::new(Memory::new(64));
        // Set X1 to a value with upper bits set; 32-bit op should ignore them.
        cpu.write_reg(1, 0xFFFF_FFFF_0000_000A); // lower 32 = 10
        cpu.write_pc(0);

        cpu.step(&mut mem).expect("step should succeed");
        // Result should be 15, zero-extended to 64 bits.
        assert_eq!(cpu.read_reg(0), 15);
        assert_eq!(cpu.read_pc(), 4);
    }

    // ── Run-loop tests ───────────────────────────────────────────────

    /// Helper: write a u32 instruction into a Cpu's internal memory.
    fn write_inst_to_cpu(cpu: &mut Cpu, addr: u64, word: u32) {
        cpu.mem.write_u32(addr, word).unwrap();
    }

    #[test]
    fn test_small_program_in_memory() {
        // Sequence:
        //   0x00: MOVZ X0, #10      → 0xD2800140  (X0 = 10)
        //   0x04: MOVZ X1, #20      → 0xD2800281  (X1 = 20)
        //   0x08: ADD X2, X1, #20   → 0x91005022  (X2 = X1 + 20 = 40)
        //   0x0C: HLT #0            → 0xD4030000

        let mut cpu = Cpu::new(Memory::new(64));
        write_inst_to_cpu(&mut cpu, 0x00, 0xD2800140); // MOVZ X0, #10
        write_inst_to_cpu(&mut cpu, 0x04, 0xD2800281); // MOVZ X1, #20
        write_inst_to_cpu(&mut cpu, 0x08, 0x91005022); // ADD X2, X1, #20
        write_inst_to_cpu(&mut cpu, 0x0C, 0xD4030000); // HLT #0

        cpu.write_pc(0x00);

        let result = cpu.run();
        assert!(result.is_ok(), "program should halt cleanly, got {:?}", result);
        assert!(cpu.halted, "CPU should be halted");

        assert_eq!(cpu.read_reg(0), 10);
        assert_eq!(cpu.read_reg(1), 20);
        assert_eq!(cpu.read_reg(2), 40);
        assert_eq!(cpu.read_pc(), 0x10); // PC advanced past HLT
    }

    #[test]
    fn test_run_halt_clean() {
        // Single HLT at PC=0.
        let mut cpu = Cpu::new(Memory::new(64));
        write_inst_to_cpu(&mut cpu, 0x00, 0xD4030000); // HLT #0
        cpu.write_pc(0x00);

        let result = cpu.run();
        assert!(result.is_ok());
        assert!(cpu.halted);
        // PC was incremented before execute, so it's 4.
        assert_eq!(cpu.read_pc(), 0x04);
    }

    #[test]
    fn test_run_unknown_instruction() {
        // 0x00000000 is not valid.
        let mut cpu = Cpu::new(Memory::new(64));
        write_inst_to_cpu(&mut cpu, 0x00, 0x0000_0000);
        cpu.write_pc(0x00);

        let result = cpu.run();
        assert_eq!(result, Err(CpuError::UnknownInstruction(0x0000_0000)));
    }

    #[test]
    fn test_print_registers_non_empty() {
        let mut cpu = Cpu::new(Memory::new(64));
        cpu.write_reg(0, 0xDEAD_BEEF);
        cpu.write_sp(0x1000);
        cpu.write_pc(0x2000);
        cpu.regs.set_flags(false, true, false, false);

        // We can't easily capture stdout, so just ensure it doesn't panic.
        cpu.print_registers();
    }
}
