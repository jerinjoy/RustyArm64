/// ARMv8-A AArch64 register file.
///
/// Contains 31 general-purpose registers (X0–X30), a dedicated stack pointer
/// (SP), program counter (PC), and four condition flags (N, Z, C, V).
///
/// In the AArch64 instruction encoding, register index 31 aliases the zero
/// register (XZR/WZR) for most source operands and the stack pointer for
/// special destination contexts.  This register file follows that convention:
///
/// * `read(31)` returns 0 (the zero register).
/// * `write(31, _)` is a no-op (write to the zero register is discarded).
/// * The stack pointer, program counter, and flags are accessed through
///   dedicated methods.
#[derive(Default)]
pub struct Registers {
    /// General-purpose registers X0–X30.
    x: [u64; 31],
    /// Stack pointer (SP / X31 in SP-aliasing contexts).
    sp: u64,
    /// Program counter.
    pc: u64,
    /// Condition flag: Negative.
    n: bool,
    /// Condition flag: Zero.
    z: bool,
    /// Condition flag: Carry.
    c: bool,
    /// Condition flag: oVerflow.
    v: bool,
}

impl Registers {
    /// Create a new register file with all registers initialised to zero
    /// and all flags cleared.
    pub fn new() -> Self {
        Self {
            x: [0u64; 31],
            sp: 0,
            pc: 0,
            n: false,
            z: false,
            c: false,
            v: false,
        }
    }

    /// Read a general-purpose register by its architectural index (0–31).
    ///
    /// * Indexes 0–30 return the corresponding `x` element.
    /// * Index 31 returns **0** (the zero register, XZR).
    ///
    /// Use [`read_sp`](Self::read_sp) and [`read_pc`](Self::read_pc) to
    /// access the stack pointer and program counter.
    ///
    /// # Panics
    /// Panics if `index > 31`.
    pub fn read(&self, index: u8) -> u64 {
        match index {
            0..=30 => self.x[index as usize],
            31 => 0, // XZR always reads as 0.
            _ => panic!("register index {} out of range (0..=31)", index),
        }
    }

    /// Write a value to a general-purpose register by its architectural
    /// index (0–31).
    ///
    /// * Indexes 0–30 write to the corresponding `x` element.
    /// * Index 31 is the zero register — writes are silently discarded.
    ///
    /// Use [`write_sp`](Self::write_sp) and [`write_pc`](Self::write_pc)
    /// to set the stack pointer and program counter.
    ///
    /// # Panics
    /// Panics if `index > 31`.
    pub fn write(&mut self, index: u8, value: u64) {
        match index {
            0..=30 => self.x[index as usize] = value,
            31 => { /* XZR – write discarded */ }
            _ => panic!("register index {} out of range (0..=31)", index),
        }
    }

    /// Read the stack pointer.
    pub fn read_sp(&self) -> u64 {
        self.sp
    }

    /// Write the stack pointer.
    pub fn write_sp(&mut self, value: u64) {
        self.sp = value;
    }

    /// Read the program counter.
    pub fn read_pc(&self) -> u64 {
        self.pc
    }

    /// Write the program counter.
    pub fn write_pc(&mut self, value: u64) {
        self.pc = value;
    }

    /// Read the four condition flags (N, Z, C, V).
    pub fn get_flags(&self) -> (bool, bool, bool, bool) {
        (self.n, self.z, self.c, self.v)
    }

    pub fn n(&self) -> bool { self.n }
    pub fn z(&self) -> bool { self.z }
    pub fn c(&self) -> bool { self.c }
    pub fn v(&self) -> bool { self.v }

    /// Set the four condition flags (N, Z, C, V).
    pub fn set_flags(&mut self, n: bool, z: bool, c: bool, v: bool) {
        self.n = n;
        self.z = z;
        self.c = c;
        self.v = v;
    }

    /// Return a human-readable multi-line dump of the entire register file,
    /// including general-purpose registers, SP, PC, and condition flags.
    pub fn dump(&self) -> String {
        let mut lines: Vec<String> = Vec::with_capacity(40);

        // Print X0–X30 in rows of 4.
        for row in 0..8 {
            let base = row * 4;
            // The last row only has 3 registers (X28–X30).
            let count = if row == 7 { 3 } else { 4 };
            let mut parts: Vec<String> = Vec::with_capacity(count);
            for col in 0..count {
                let idx = base + col;
                parts.push(format!("X{:02}=0x{:016X}", idx, self.x[idx]));
            }
            lines.push(parts.join("  "));
        }

        lines.push(format!("SP =0x{:016X}", self.sp));
        lines.push(format!("PC =0x{:016X}", self.pc));
        lines.push(format!(
            "N={} Z={} C={} V={}",
            self.n as u8, self.z as u8, self.c as u8, self.v as u8
        ));

        lines.join("\n")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_zero() {
        let regs = Registers::new();
        for i in 0..31 {
            assert_eq!(regs.x[i], 0, "x[{}] should be 0", i);
        }
        assert_eq!(regs.sp, 0);
        assert_eq!(regs.pc, 0);
        assert!(!regs.n);
        assert!(!regs.z);
        assert!(!regs.c);
        assert!(!regs.v);
    }

    #[test]
    fn test_read_write_gpr() {
        let mut regs = Registers::new();

        // Write and read back X0–X30.
        for i in 0..31u8 {
            let val = 0xDEAD_0000_0000_0000u64 + i as u64;
            regs.write(i, val);
            let got = regs.read(i);
            assert_eq!(got, val, "mismatch for register index {}", i);
        }

        // Register 31 (XZR): write is discarded, read returns 0.
        regs.write(31, 0xCAFE);
        assert_eq!(regs.read(31), 0);
    }

    #[test]
    fn test_sp_pc_access() {
        let mut regs = Registers::new();

        regs.write_sp(0x1234_5678_9ABC_DEF0);
        assert_eq!(regs.read_sp(), 0x1234_5678_9ABC_DEF0);

        regs.write_pc(0x8000_0000);
        assert_eq!(regs.read_pc(), 0x8000_0000);

        // Ensure SP and PC are independent of the GPR array.
        assert_eq!(regs.read(31), 0); // XZR still reads 0.
        regs.write_sp(0);
        regs.write_pc(0);
        for i in 0..31 {
            // X0–X30 should still hold their values from the previous test
            // or be zero if this is the only test touching them.
            // Actually in this test they were never written, so:
            assert_eq!(regs.x[i], 0);
        }
    }

    #[test]
    fn test_flag_get_set() {
        let mut regs = Registers::new();

        // All flags initially false.
        assert!(!regs.n);
        assert!(!regs.z);
        assert!(!regs.c);
        assert!(!regs.v);

        regs.set_flags(true, true, false, true);
        assert!(regs.n);
        assert!(regs.z);
        assert!(!regs.c);
        assert!(regs.v);

        regs.set_flags(false, false, false, false);
        assert!(!regs.n);
        assert!(!regs.z);
        assert!(!regs.c);
        assert!(!regs.v);
    }

    #[test]
    fn test_dump_non_empty() {
        let mut regs = Registers::new();
        regs.x[0] = 0xDEAD_BEEF;
        regs.sp = 0x1000;
        regs.pc = 0x2000;
        regs.set_flags(false, true, false, false);

        let dump = regs.dump();
        assert!(dump.contains("X00=0x00000000DEADBEEF"));
        assert!(dump.contains("SP =0x0000000000001000"));
        assert!(dump.contains("PC =0x0000000000002000"));
        assert!(dump.contains("Z=1"));
    }

    #[test]
    fn test_default_trait() {
        let regs: Registers = Default::default();
        assert_eq!(regs.read_pc(), 0);
        assert_eq!(regs.read_sp(), 0);
        assert!(!regs.n);
    }
}
