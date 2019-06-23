# modtype

[![Build Status](https://img.shields.io/travis/com/qryxip/modtype/master.svg?label=windows%20%26%20macos%20%26%20linux)](https://travis-ci.com/qryxip/modtype)
[![codecov](https://codecov.io/gh/qryxip/modtype/branch/master/graph/badge.svg)](https://codecov.io/gh/qryxip/modtype)
[![Crates.io](https://img.shields.io/crates/v/modtype.svg)](https://crates.io/crates/modtype)

This crate provides:
- Macros that implement modular arithmetic integer types
- Preset types
    - [`modtype::preset::u64::F`]
    - [`modtype::preset::u64::Z`]
    - [`modtype::preset::u64::thread_local::F`]
    - [`modtype::preset::u64::mod1000000007::F`]
    - [`modtype::preset::u64::mod1000000007::Z`]

## Usage

```rust
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
```

```rust
use modtype::preset::u64::mod1000000007::{F, Z};
```

## Requirements

- The inner value is [`u8`], [`u16`], [`u32`], [`u64`], [`u128`], or [`usize`].
- The inner value and the modulus are of a same type.
- The modulus is immutable.
- The inner value is always smaller than the modulus.
    - If the modular arithmetic type implements [`One`], The modulus is larger than `1`.
- If the modular arithmetic type implements [`Div`], the modulus is a prime.

### Struct

| Name                 | Format                                                                   | Optional                         |
| :------------------- | :----------------------------------------------------------------------- | :------------------------------- |
| `modulus`            | `modulus = #`[`Lit`] where `#`[`Lit`] is converted/parsed to an [`Expr`] | No                               |
| `std`                | `std = #`[`LitStr`] where `#`[`LitStr`] is parsed to a [`Path`]          | Yes (default = `::std`)          |
| `num_traits`         | `num_traits = #`[`LitStr`] where `#`[`LitStr`] is parsed to a [`Path`]   | Yes (default = `::num::traits`)  |
| `num_integer`        | `num_integer = #`[`LitStr`] where `#`[`LitStr`] is parsed to a [`Path`]  | Yes (default = `::num::integer`) |
| `num_bigint`         | `num_bigint = #`[`LitStr`] where `#`[`LitStr`] is parsed to a [`Path`]   | Yes (default = `::num::bigint`)  |
| `no_impl_for_ref`    | `no_impl_for_ref`                                                        | Yes                              |

### Field

| Name                 | Format  | Optional |
| :------------------- | :------ | :------- |
| `value`              | `value` | No       |

### [`ConstValue`]

#### Struct

| Name                 | Format                                                       | Optional  |
| :------------------- | :----------------------------------------------------------- | :-------- |
| `const_value`        | `const_value = #`[`LitInt`] where `#`[`LitInt`] has a suffix | No        |

[`u8`]: https://doc.rust-lang.org/nightly/std/primitive.u8.html
[`u16`]: https://doc.rust-lang.org/nightly/std/primitive.u16.html
[`u32`]: https://doc.rust-lang.org/nightly/std/primitive.u32.html
[`u64`]: https://doc.rust-lang.org/nightly/std/primitive.u64.html
[`u128`]: https://doc.rust-lang.org/nightly/std/primitive.u128.html
[`usize`]: https://doc.rust-lang.org/nightly/std/primitive.usize.html
[`Div`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Div.html
[`One`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.One.html
[`Lit`]: https://docs.rs/syn/0.15/syn/enum.Lit.html
[`LitStr`]: https://docs.rs/syn/0.15/syn/struct.LitStr.html
[`LitInt`]: https://docs.rs/syn/0.15/syn/struct.LitInt.html
[`Expr`]: https://docs.rs/syn/0.15/syn/struct.Expr.html
[`Path`]: https://docs.rs/syn/0.15/syn/struct.Path.html
[`ConstValue`]: https://docs.rs/modtype_derive/0.3/modtype_derive/derive.ConstValue.html
[`modtype::preset::u64::F`]: https://docs.rs/modtype/0.3/modtype/preset/u64/struct.F.html
[`modtype::preset::u64::Z`]: https://docs.rs/modtype/0.3/modtype/preset/u64/struct.Z.html
[`modtype::preset::u64::thread_local::F`]: https://docs.rs/modtype/0.3/modtype/preset/u64/thread_local/struct.F.html
[`modtype::preset::u64::mod1000000007::F`]: https://docs.rs/modtype/0.3/modtype/preset/u64/mod1000000007/type.F.html
[`modtype::preset::u64::mod1000000007::Z`]: https://docs.rs/modtype/0.3/modtype/preset/u64/mod1000000007/type.Z.html
