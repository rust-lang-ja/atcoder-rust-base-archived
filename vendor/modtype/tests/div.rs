use num::traits::Inv as _;

#[test]
fn test_div_for_mod17() {
    #[derive(
        Clone, Copy, modtype::From, modtype::Deref, modtype::Mul, modtype::Div, modtype::new,
    )]
    #[modtype(modulus = 17)]
    struct F {
        #[modtype(value)]
        __value: u32,
    }

    for a in 0..=16 {
        for b in 1..=16 {
            assert_eq!(*(F::new(a) / F::new(b) * F::new(b)), a);
        }
    }
}

#[test]
fn test_div_for_mod1000000007() {
    use modtype::preset::u64::mod1000000007::F;

    assert_eq!(F::new(13).inv(), F::new(153_846_155));
}
