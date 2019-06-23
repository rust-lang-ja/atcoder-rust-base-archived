use modtype::preset::u64::mod1000000007::Z;

fn main() {
    static INPUT: &str = "13";
    let mut a = INPUT.parse::<Z>().unwrap();
    a += Z::new(1_000_000_000);
    dbg!(a);
}
