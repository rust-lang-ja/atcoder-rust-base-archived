use modtype::ConstValue;

use std::marker::PhantomData;

fn main() {
    println!(
        "{} / {} ≡ {}",
        F::new(3),
        F::new(4),
        F::new(3) / F::new(4),
    );
}

type F = F_<Const17U32>;

#[derive(
    Default,
    Clone,
    Copy,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    modtype::From,
    modtype::Into,
    modtype::FromStr,
    modtype::Display,
    modtype::Debug,
    modtype::Deref,
    modtype::Neg,
    modtype::Add,
    modtype::AddAssign,
    modtype::Sub,
    modtype::SubAssign,
    modtype::Mul,
    modtype::MulAssign,
    modtype::Div,
    modtype::DivAssign,
    modtype::Rem,
    modtype::RemAssign,
    modtype::Zero,
    modtype::One,
    modtype::Num,
    modtype::Bounded,
    modtype::CheckedAdd,
    modtype::CheckedSub,
    modtype::CheckedMul,
    modtype::CheckedDiv,
    modtype::CheckedRem,
    modtype::CheckedNeg,
    modtype::Inv,
    modtype::Unsigned,
    modtype::FromPrimitive,
    modtype::ToPrimitive,
    modtype::Pow_u8,
    modtype::Pow_u16,
    modtype::Pow_u32,
    modtype::Pow_usize,
    modtype::Integer,
    modtype::ToBigInt,
    modtype::ToBigUint,
    modtype::new,
    modtype::get,
)]
#[modtype(
    modulus = "M::VALUE",
    std = "std",
    num_traits = "num::traits",
    num_integer = "num::integer",
    num_bigint = "num::bigint",
    no_impl_for_ref
)]
struct F_<M: ConstValue<Value = u32>> {
    #[modtype(value)]
    __value: u32,
    phantom: PhantomData<fn() -> M>,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, ConstValue)]
#[modtype(const_value = 17u32)]
enum Const17U32 {}
