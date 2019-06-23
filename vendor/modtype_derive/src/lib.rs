#![recursion_limit = "256"]

extern crate proc_macro;

use if_chain::if_chain;
use proc_macro2::Span;
use quote::quote;
use syn::spanned::Spanned;
use syn::{
    parse_macro_input, parse_quote, Attribute, BinOp, Data, DataEnum, DataStruct, DataUnion,
    DeriveInput, Expr, ExprStruct, Field, Fields, FieldsNamed, Generics, Ident, ImplGenerics,
    IntSuffix, ItemFn, Lit, Meta, MetaList, MetaNameValue, NestedMeta, Path, Type, Visibility,
};

use std::convert::TryFrom;
use std::{fmt, mem};

macro_rules! try_syn {
    ($expr:expr) => {
        match $expr {
            Ok(expr) => expr,
            Err::<_, syn::Error>(err) => return err.to_compile_error().into(),
        }
    };
}

/// Derives [`ConstValue`].
///
/// # Attributes
///
/// ## Struct
///
/// | Name                 | Format                                                       | Optional  |
/// | :------------------- | :----------------------------------------------------------- | :-------- |
/// | `const_value`        | `const_value = #`[`LitInt`] where `#`[`LitInt`] has a suffix | No        |
///
/// [`ConstValue`]: https://docs.rs/modtype/0.3/modtype/trait.ConstValue.html
/// [`LitInt`]: https://docs.rs/syn/0.15/syn/struct.LitInt.html
#[proc_macro_derive(ConstValue, attributes(modtype))]
pub fn const_value(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fn compile_error(span: Span, msg: &str) -> proc_macro::TokenStream {
        syn::Error::new(span, msg).to_compile_error().into()
    }

    let DeriveInput {
        attrs,
        ident,
        generics,
        ..
    } = parse_macro_input!(input as DeriveInput);

    if !generics.params.is_empty() {
        return compile_error(generics.span(), "The generics parameters must be empty");
    }

    static MSG: &str = "expected `modtype(const_value = #LitInt)` where the `LitInt` has a suffix";

    let mut int = None;

    for attr in &attrs {
        if let Ok(meta) = attr.parse_meta() {
            match &meta {
                Meta::Word(ident) | Meta::NameValue(MetaNameValue { ident, .. })
                    if ident == "modtype" =>
                {
                    return compile_error(ident.span(), MSG);
                }
                Meta::List(MetaList { ident, nested, .. }) if ident == "modtype" => {
                    let (value, ty, span) = if_chain! {
                        if nested.len() == 1;
                        if let NestedMeta::Meta(Meta::NameValue(name_value)) = &nested[0];
                        if name_value.ident == "const_value";
                        if let Lit::Int(int) = &name_value.lit;
                        if let Some::<Type>(ty) = match int.suffix() {
                            IntSuffix::I8 => Some(parse_quote!(i8)),
                            IntSuffix::I16 => Some(parse_quote!(i16)),
                            IntSuffix::I32 => Some(parse_quote!(i32)),
                            IntSuffix::I64 => Some(parse_quote!(i64)),
                            IntSuffix::I128 => Some(parse_quote!(i128)),
                            IntSuffix::Isize => Some(parse_quote!(isize)),
                            IntSuffix::U8 => Some(parse_quote!(u8)),
                            IntSuffix::U16 => Some(parse_quote!(u16)),
                            IntSuffix::U32 => Some(parse_quote!(u32)),
                            IntSuffix::U64 => Some(parse_quote!(u64)),
                            IntSuffix::U128 => Some(parse_quote!(u128)),
                            IntSuffix::Usize => Some(parse_quote!(usize)),
                            IntSuffix::None => None,
                        };
                        then {
                            (int.clone(), ty, ident.span())
                        } else {
                            return compile_error(ident.span(), MSG);
                        }
                    };
                    if mem::replace(&mut int, Some((value, ty))).is_some() {
                        return compile_error(span, "multiple definition");
                    }
                }
                _ => {}
            }
        }
    }

    let (int, ty) = match int {
        None => return compile_error(Span::call_site(), MSG),
        Some(int) => int,
    };

    quote!(
        impl ::modtype::ConstValue for #ident {
            type Value = #ty;
            const VALUE: #ty = #int;
        }
    )
    .into()
}

/// Derives [`From`]`<#InnerValue>`.
///
/// # Requirements
///
/// - The fields are [`Default`] except `#InnerValue`.
///
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
#[proc_macro_derive(From, attributes(modtype))]
pub fn from(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        modulus,
        std,
        struct_ident,
        generics,
        field_ty,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let struct_expr = input.construct_self(true, Some(parse_quote!(from % #modulus)));

    quote!(
        impl #impl_generics #std::convert::From<#field_ty> for #struct_ident#ty_generics
        #where_clause
        {
            #[inline]
            fn from(from: #field_ty) -> Self {
                #struct_expr
            }
        }
    )
    .into()
}

/// Derives [`From`]`<Self> for #InnerValue`.
///
/// # Requirements
///
/// - `#InnerValue` is not a type parameter.
///
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(Into, attributes(modtype))]
pub fn into(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl #impl_generics #std::convert::From<#struct_ident#ty_generics> for #field_ty
        #where_clause
        {
            #[inline]
            fn from(from: #struct_ident#ty_generics) -> Self {
                from.#field_ident
            }
        }
    )
    .into()
}

/// Derives [`FromStr`]`<Err = #InnerValue::Err>`.
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`FromStr`]: https://doc.rust-lang.org/nightly/core/str/trait.FromStr.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(FromStr, attributes(modtype))]
pub fn from_str(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        struct_ident,
        generics,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #std::str::FromStr for #struct_ident#ty_generics
        #where_clause
        {
            type Err = <#field_ty as #std::str::FromStr>::Err;

            #[inline]
            fn from_str(s: &str) -> #std::result::Result<Self, <#field_ty as #std::str::FromStr>::Err> {
                let value = <#field_ty as #std::str::FromStr>::from_str(s)?;
                Ok(<Self as #std::convert::From<#field_ty>>::from(value))
            }
        }
    )
    .into()
}

/// Derives [`Display`].
///
/// # Requirements
///
/// - `#InnerValue: `[`Display`].
///
/// [`Display`]: https://doc.rust-lang.org/nightly/core/fmt/trait.Display.html
#[proc_macro_derive(Display, attributes(modtype))]
pub fn display(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fmt(input, parse_quote!(Display))
}

/// Derives [`Debug`].
///
/// # Requirements
///
/// - `#InnerValue: `[`Debug`].
///
/// [`Debug`]: https://doc.rust-lang.org/nightly/core/fmt/trait.Debug.html
#[proc_macro_derive(DebugTransparent, attributes(modtype))]
pub fn debug_transparent(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    fmt(input, parse_quote!(Debug))
}

/// Derives [`Deref`]`<Target = #InnerValue>`.
///
/// # Requirements
///
/// Nothing.
///
/// [`Deref`]: https://doc.rust-lang.org/nightly/core/ops/deref/trait.Deref.html
#[proc_macro_derive(Deref, attributes(modtype))]
pub fn deref(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #std::ops::Deref for #struct_ident#ty_generics
        #where_clause
        {
            type Target = #field_ty;

            #[inline]
            fn deref(&self) -> &#field_ty {
                &self.#field_ident
            }
        }
    )
    .into()
}

/// Derives [`Neg`]`.
///
/// # Requirements
///
/// - `Self: `[`Add`]`<Self, Output = Self>`
/// - `Self: `[`Sub`]`<Self, Output = Self>`
/// - The fields are [`Default`] except `#InnerValue`.
///
/// [`Neg`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Neg.html
/// [`Add`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Add.html
/// [`Sub`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Sub.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
#[proc_macro_derive(Neg, attributes(modtype))]
pub fn neg(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        modulus,
        std,
        no_impl_for_ref,
        struct_ident,
        generics,
        field_ident,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let value_expr = parse_quote!(#modulus - self.#field_ident);
    let struct_expr = input.construct_self(false, Some(value_expr));

    let derive = |lhs_ty: Type| {
        quote! {
            impl#impl_generics #std::ops::Neg for #lhs_ty
            #where_clause
            {
                type Output = #struct_ident#ty_generics;

                #[inline]
                fn neg(self) -> #struct_ident#ty_generics {
                    fn static_assert_add<O, L: #std::ops::Add<L, Output = O>>() {}
                    fn static_assert_sub<O, L: #std::ops::Sub<L, Output = O>>() {}
                    static_assert_add::<#struct_ident#ty_generics, Self>();
                    static_assert_sub::<#struct_ident#ty_generics, Self>();
                    #struct_expr
                }
            }
        }
    };

    let mut ret = derive(parse_quote!(#struct_ident#ty_generics));
    if !no_impl_for_ref {
        ret.extend(derive(parse_quote!(&'_ #struct_ident#ty_generics)))
    }
    ret.into()
}

/// Derives [`Add`].
///
/// # Requirements
///
/// - The fields are [`Default`] except `#InnerValue`.
///
/// [`Add`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Add.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
#[proc_macro_derive(Add, attributes(modtype))]
pub fn add(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_almost_transparent(
        input,
        parse_quote!(Add),
        parse_quote!(add),
        |l, r, _| parse_quote!(#l + #r),
    )
}

/// Derives [`AddAssign`]`.
///
/// # Requirements
///
/// - `Self: `[`Add`]`<Self, Output = Self>`.
/// - `Self: Copy`.
///
/// [`AddAssign`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.AddAssign.html
/// [`Add`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Add.html
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
#[proc_macro_derive(AddAssign, attributes(modtype))]
pub fn add_assign(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_assign(
        input,
        parse_quote!(AddAssign),
        parse_quote!(add_assign),
        parse_quote!(+),
    )
}

/// Derives [`Sub`]`.
///
/// # Requirements
///
/// - The fields are [`Default`] except `#InnerValue`.
///
/// [`Sub`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Sub.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
#[proc_macro_derive(Sub, attributes(modtype))]
pub fn sub(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_almost_transparent(
        input,
        parse_quote!(Sub),
        parse_quote!(sub),
        |l, r, m| parse_quote!(#m + #l - #r),
    )
}

/// Derives [`SubAssign`]`.
///
/// # Requirements
///
/// - `Self: `[`Sub`]`<Self, Output = Self>`
/// - `Self: `[`Copy`]
///
/// [`SubAssign`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.SubAssign.html
/// [`Sub`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Sub.html
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
#[proc_macro_derive(SubAssign, attributes(modtype))]
pub fn sub_assign(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_assign(
        input,
        parse_quote!(SubAssign),
        parse_quote!(sub_assign),
        parse_quote!(-),
    )
}

/// Derives [`Mul`]`.
///
/// # Requirements
///
/// - The fields are [`Default`] except `#InnerValue`.
///
/// [`Mul`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Mul.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
#[proc_macro_derive(Mul, attributes(modtype))]
pub fn mul(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_almost_transparent(
        input,
        parse_quote!(Mul),
        parse_quote!(mul),
        |l, r, _| parse_quote!(#l * #r),
    )
}

/// Derives [`MulAssign`]`.
///
/// # Requirements
///
/// - `Self: `[`Mul`]`<Self, Output = Self>`.
/// - `Self: `[`Copy`].
///
/// [`MulAssign`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.MulAssign.html
/// [`Mul`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Mul.html
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
#[proc_macro_derive(MulAssign, attributes(modtype))]
pub fn mul_assign(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_assign(
        input,
        parse_quote!(MulAssign),
        parse_quote!(mul_assign),
        parse_quote!(*),
    )
}

/// Derives [`Div`]`.
///
/// # Requirements
///
/// - `<#InnerValue as `[`ToPrimitive`]`>::`[`to_i128`] always return [`Some`] for values in [0, `#modulus`).
/// - `#modulus` is a prime.
/// - The fields are [`Default`] except `#InnerValue`.
///
/// [`Div`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Div.html
/// [`ToPrimitive`]: https://docs.rs/num-traits/0.2/num_traits/cast/trait.ToPrimitive.html
/// [`to_i128`]: https://docs.rs/num-traits/0.2/num_traits/cast/trait.ToPrimitive.html#method.to_i128
/// [`Some`]: https://doc.rust-lang.org/nightly/core/option/enum.Option.html#variant.Some
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
#[proc_macro_derive(Div, attributes(modtype))]
pub fn div(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin(input, parse_quote!(Div), |input, rhs_ty| {
        let Input {
            modulus,
            std,
            num_traits,
            struct_ident,
            generics,
            field_ident,
            field_ty,
            ..
        } = input;
        let (_, ty_generics, _) = generics.split_for_impl();

        let struct_expr = input.construct_self(false, None);

        parse_quote! {
            fn div(self, rhs: #rhs_ty) -> #struct_ident#ty_generics {
                fn extended_gcd(a: i128, b: i128) -> (i128, i128, i128) {
                    if a == 0 {
                        (b, 0, 1)
                    } else {
                        let (d, u, v) = extended_gcd(b % a, a);
                        (d, v - (b / a) * u, u)
                    }
                }

                let lhs = <#field_ty as #num_traits::ToPrimitive>::to_i128(&self.#field_ident);
                let lhs = #std::option::Option::expect(lhs, "could not convert to `i128`");
                let rhs = <#field_ty as #num_traits::ToPrimitive>::to_i128(&rhs.#field_ident);
                let rhs = #std::option::Option::expect(rhs, "could not convert to `i128`");
                let modulus = <#field_ty as #num_traits::ToPrimitive>::to_i128(&#modulus);
                let modulus = #std::option::Option::expect(modulus, "could not convert to `i128`");
                if rhs == 0 {
                    panic!("attempt to divide by zero");
                }
                let (d, u, _) = extended_gcd(rhs, modulus);
                if rhs % d != 0 {
                    panic!("RHS is not a multiple of gcd({}, {})", rhs, modulus);
                }
                let mut #field_ident = (lhs * u) % modulus;
                if #field_ident < 0 {
                    #field_ident += modulus;
                }
                let #field_ident = <#field_ty as #num_traits::FromPrimitive>::from_i128(#field_ident);
                let #field_ident = #std::option::Option::unwrap(#field_ident);
                #struct_expr
            }
        }
    })
}

/// Derives [`DivAssign`]`.
///
/// # Requirements
///
/// - `Self: `[`Div`]`<Self, Output = Self>`.
/// - `Self: `[`Copy`].
///
/// [`DivAssign`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.DivAssign.html
/// [`Div`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Div.html
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
#[proc_macro_derive(DivAssign, attributes(modtype))]
pub fn div_assign(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_assign(
        input,
        parse_quote!(DivAssign),
        parse_quote!(div_assign),
        parse_quote!(/),
    )
}

/// Derives [`Rem`]`.
///
/// # Requirements
///
/// - `Self: `[`Div`]`<Self, Output = Self>`.
/// - `Self: `[`Zero`].
///
/// [`Rem`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Rem.html
/// [`Div`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Div.html
/// [`Zero`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.Zero.html
#[proc_macro_derive(Rem, attributes(modtype))]
pub fn rem(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin(input, parse_quote!(Rem), |input, rhs_ty| {
        let Input {
            std,
            num_traits,
            struct_ident,
            generics,
            ..
        } = input;
        let (_, ty_generics, _) = generics.split_for_impl();

        parse_quote! {
            fn rem(self, rhs: #rhs_ty) -> #struct_ident#ty_generics {
                fn static_assert_div<T: #std::ops::Div<T, Output = T>>() {}
                fn static_assert_zero<T: #num_traits::Zero>() {}
                static_assert_div::<#struct_ident#ty_generics>();
                static_assert_zero::<#struct_ident#ty_generics>();

                if <#struct_ident#ty_generics as #num_traits::Zero>::is_zero(&rhs) {
                    panic!("attempt to calculate the remainder with a divisor of zero")
                }
                <#struct_ident#ty_generics as #num_traits::Zero>::zero()
            }
        }
    })
}

/// Derives [`RemAssign`]`.
///
/// # Requirements
///
/// - `Self: `[`Rem`]`<Self, Output = Self>`.
/// - `Self: `[`Copy`].
///
/// [`RemAssign`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.RemAssign.html
/// [`Rem`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Rem.html
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
#[proc_macro_derive(RemAssign, attributes(modtype))]
pub fn rem_assign(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    bin_assign(
        input,
        parse_quote!(RemAssign),
        parse_quote!(rem_assign),
        parse_quote!(%),
    )
}

/// Derives [`Zero`].
///
/// # Requirements
///
/// - The fields are [`Default`] except `#InnerValue`.
/// - `Self: `[`Add`]`<Self, Output = Self>`. (required by [`Zero`] itself)
///
/// [`Zero`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.Zero.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
/// [`Add`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Add.html
#[proc_macro_derive(Zero, attributes(modtype))]
pub fn zero(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    identity(
        input,
        parse_quote!(Zero),
        parse_quote!(zero),
        parse_quote!(is_zero),
    )
}

/// Derives [`One`].
///
/// # Requirements
///
/// - The fields are [`Default`] except `#InnerValue`.
/// - `Self: `[`Mul`]`<Self, Output = Self>`. (required by [`One`] itself)
/// - `Self: `[`PartialEq`]`<Self>`. (required by `One::is_one`)
///
/// [`One`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.One.html
/// [`Default`]: https://doc.rust-lang.org/nightly/core/default/trait.Default.html
/// [`Mul`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Mul.html
/// [`PartialEq`]: https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html
#[proc_macro_derive(One, attributes(modtype))]
pub fn one(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    identity(
        input,
        parse_quote!(One),
        parse_quote!(one),
        parse_quote!(is_one),
    )
}

/// Derives [`Num`].
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
/// - `Self: `[`Zero`]. (required by [`Num`] itself)
/// - `Self: `[`One`]. (required by [`Num`] itself)
/// - `Self: `[`NumOps`]`<Self, Self>`. (required by [`Num`] itself)
/// - `Self: `[`PartialEq`]`<Self>`. (required by [`Num`] itself)
///
/// [`Num`]: https://docs.rs/num-traits/0.2/num_traits/trait.Num.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
/// [`Zero`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.Zero.html
/// [`One`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.One.html
/// [`NumOps`]: https://docs.rs/num-traits/0.2/num_traits/trait.NumOps.html
/// [`PartialEq`]: https://doc.rust-lang.org/nightly/core/cmp/trait.PartialEq.html
#[proc_macro_derive(Num, attributes(modtype))]
pub fn num(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        num_traits,
        struct_ident,
        generics,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #num_traits::Num for #struct_ident#ty_generics
        #where_clause
        {
            type FromStrRadixErr = <#field_ty as #num_traits::Num>::FromStrRadixErr;

            #[inline]
            fn from_str_radix(str: &str, radix: u32) -> #std::result::Result<Self, <#field_ty as #num_traits::Num>::FromStrRadixErr> {
                let value = <#field_ty as #num_traits::Num>::from_str_radix(str, radix)?;
                Ok(<Self as #std::convert::From<#field_ty>>::from(value))
            }
        }
    )
    .into()
}

/// Derives [`Bounded`].
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`Bounded`]: https://docs.rs/num-traits/0.2/num_traits/bounds/trait.Bounded.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(Bounded, attributes(modtype))]
pub fn bounded(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        modulus,
        std,
        num_traits,
        struct_ident,
        generics,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #num_traits::Bounded for #struct_ident#ty_generics
        #where_clause
        {
            #[inline]
            fn min_value() -> Self {
                let zero = <#field_ty as #num_traits::Zero>::zero();
                <Self as #std::convert::From<#field_ty>>::from(zero)
            }

            #[inline]
            fn max_value() -> Self {
                let minus_1 = #modulus - <#field_ty as #num_traits::One>::one();
                <Self as #std::convert::From<#field_ty>>::from(minus_1)
            }
        }
    )
    .into()
}

/// Derives [`CheckedAdd`].
///
/// # Requirements
///
/// - `Self: `[`Copy`].
/// - `Self: `[`Add`]`<Self, Output = Self>`. (required by [`CheckedAdd`] itself)
///
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Add`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Add.html
/// [`CheckedAdd`]: https://docs.rs/num-traits/0.2/num_traits/ops/checked/trait.CheckedAdd.html
#[proc_macro_derive(CheckedAdd, attributes(modtype))]
pub fn checked_add(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    checked_bin(
        input,
        parse_quote!(CheckedAdd),
        parse_quote!(checked_add),
        false,
        parse_quote!(+),
    )
}

/// Derives `CheckedSub`.
///
/// # Requirements
///
/// - `Self: `[`Copy`].
/// - `Self: `[`Sub`]`<Self, Output = Self>`. (required by [`CheckedSub`] itself)
///
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Sub`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Sub.html
/// [`CheckedAdd`]: https://docs.rs/num-traits/0.2/num_traits/ops/checked/trait.CheckedAdd.html
#[proc_macro_derive(CheckedSub, attributes(modtype))]
pub fn checked_sub(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    checked_bin(
        input,
        parse_quote!(CheckedSub),
        parse_quote!(checked_sub),
        false,
        parse_quote!(-),
    )
}

/// Derives [`CheckedMul`].
///
/// # Requirements
///
/// - `Self: `[`Copy`].
/// - `Self: `[`Mul`]`<Self, Output = Self>`. (required by [`CheckedMul`] itself)
///
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Mul`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Mul.html
/// [`CheckedMul`]: https://docs.rs/num-traits/0.2/num_traits/ops/checked/trait.CheckedMul.html
#[proc_macro_derive(CheckedMul, attributes(modtype))]
pub fn checked_mul(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    checked_bin(
        input,
        parse_quote!(CheckedMul),
        parse_quote!(checked_mul),
        false,
        parse_quote!(*),
    )
}

/// Derives [`CheckedDiv`].
///
/// # Requirements
///
/// - `Self: `[`Copy`].
/// - `Self: `[`Div`]`<Self, Output = Self>`. (required by [`CheckedDiv`] itself)
///
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Div`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Div.html
/// [`CheckedDiv`]: https://docs.rs/num-traits/0.2/num_traits/ops/checked/trait.CheckedDiv.html
#[proc_macro_derive(CheckedDiv, attributes(modtype))]
pub fn checked_div(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    checked_bin(
        input,
        parse_quote!(CheckedDiv),
        parse_quote!(checked_div),
        true,
        parse_quote!(/),
    )
}

/// Derives [`CheckedRem`].
///
/// # Requirements
///
/// - `Self: `[`Copy`].
/// - `Self: `[`Rem`]`<Self, Output = Self>`. (required by [`CheckedRem`] itself)
///
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Rem`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Rem.html
/// [`CheckedRem`]: https://docs.rs/num-traits/0.2/num_traits/ops/checked/trait.CheckedRem.html
#[proc_macro_derive(CheckedRem, attributes(modtype))]
pub fn checked_rem(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    checked_bin(
        input,
        parse_quote!(CheckedRem),
        parse_quote!(checked_rem),
        true,
        parse_quote!(%),
    )
}

/// Derives [`CheckedNeg`].
///
/// # Requirements
///
/// - `Self: `[`Copy`].
/// - `Self: `[`Neg`]`<Output = Self>`.
///
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Neg`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Neg.html
/// [`CheckedNeg`]: https://docs.rs/num-traits/0.2/num_traits/ops/checked/trait.CheckedNeg.html
#[proc_macro_derive(CheckedNeg, attributes(modtype))]
pub fn checked_neg(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        num_traits,
        struct_ident,
        generics,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #num_traits::CheckedNeg for #struct_ident#ty_generics
            #where_clause
        {
            #[inline]
            fn checked_neg(&self) -> #std::option::Option<Self> {
                fn static_assert_copy<T: #std::marker::Copy>() {}
                static_assert_copy::<Self>();
                Some(-*self)
            }
        }
    )
    .into()
}

/// Derives [`Inv`].
///
/// # Requirements
///
/// - `Self: `[`One`].
/// - `Self: `[`Div`]`<Self, Output = Self>`.
///
/// [`Inv`]: https://docs.rs/num-traits/0.2/num_traits/ops/inv/trait.Inv.html
/// [`One`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.One.html
/// [`Div`]: https://doc.rust-lang.org/nightly/core/ops/arith/trait.Div.html
#[proc_macro_derive(Inv, attributes(modtype))]
pub fn inv(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        num_traits,
        no_impl_for_ref,
        struct_ident,
        generics,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let derive = |ty: Type| -> _ {
        quote! {
            impl#impl_generics #num_traits::Inv for #ty
                #where_clause
            {
                type Output = #struct_ident#ty_generics;

                #[inline]
                fn inv(self) -> #struct_ident#ty_generics {
                    <#struct_ident#ty_generics as #num_traits::One>::one() / self
                }
            }
        }
    };

    let mut ret = derive(parse_quote!(#struct_ident#ty_generics));
    if !no_impl_for_ref {
        ret.extend(derive(parse_quote!(&'_ #struct_ident#ty_generics)));
    }
    ret.into()
}

/// Derives [`Unsigned`].
///
/// # Requirements
///
/// - `Self: `[`Num`]. (required by [`Unsigned`] itself)
///
/// [`Unsigned`]: https://docs.rs/num-traits/0.2/num_traits/sign/trait.Unsigned.html
/// [`Num`]: https://docs.rs/num-traits/0.2/num_traits/trait.Num.html
#[proc_macro_derive(Unsigned, attributes(modtype))]
pub fn unsigned(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        num_traits,
        struct_ident,
        generics,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #num_traits::Unsigned for #struct_ident#ty_generics
        #where_clause {}
    )
    .into()
}

/// Derives [`FromPrimitive`].
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`FromPrimitive`]: https://docs.rs/num-traits/0.2/num_traits/cast/trait.FromPrimitive.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(FromPrimitive, attributes(modtype))]
pub fn from_primitive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        num_traits,
        struct_ident,
        generics,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let derive = |prim: Ident| -> ItemFn {
        let fn_ident = Ident::new(&format!("from_{}", prim), Span::call_site());
        parse_quote! {
            #[inline]
            fn #fn_ident(n: #prim) -> Option<Self> {
                <#field_ty as #num_traits::FromPrimitive>::#fn_ident(n)
                    .map(<Self as #std::convert::From<#field_ty>>::from)
            }
        }
    };

    let methods = vec![
        derive(parse_quote!(i64)),
        derive(parse_quote!(u64)),
        derive(parse_quote!(isize)),
        derive(parse_quote!(i8)),
        derive(parse_quote!(i16)),
        derive(parse_quote!(i32)),
        derive(parse_quote!(i128)),
        derive(parse_quote!(usize)),
        derive(parse_quote!(u8)),
        derive(parse_quote!(u16)),
        derive(parse_quote!(u32)),
        derive(parse_quote!(u128)),
        derive(parse_quote!(f32)),
        derive(parse_quote!(f64)),
    ];

    quote!(
        impl#impl_generics #num_traits::FromPrimitive for #struct_ident#ty_generics
        #where_clause
        {
            #(#methods)*
        }
    )
    .into()
}

/// Derives [`ToPrimitive`].
///
/// # Requirements
///
/// Nothing.
///
/// [`ToPrimitive`]: https://docs.rs/num-traits/0.2/num_traits/cast/trait.ToPrimitive.html
#[proc_macro_derive(ToPrimitive, attributes(modtype))]
pub fn to_primitive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        num_traits,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let derive = |prim: Ident| -> ItemFn {
        let fn_ident = Ident::new(&format!("to_{}", prim), Span::call_site());
        parse_quote! {
            #[inline]
            fn #fn_ident(&self) -> Option<#prim> {
                <#field_ty as #num_traits::ToPrimitive>::#fn_ident(&self.#field_ident)
            }
        }
    };

    let methods = vec![
        derive(parse_quote!(i64)),
        derive(parse_quote!(u64)),
        derive(parse_quote!(isize)),
        derive(parse_quote!(i8)),
        derive(parse_quote!(i16)),
        derive(parse_quote!(i32)),
        derive(parse_quote!(i128)),
        derive(parse_quote!(usize)),
        derive(parse_quote!(u8)),
        derive(parse_quote!(u16)),
        derive(parse_quote!(u32)),
        derive(parse_quote!(u128)),
        derive(parse_quote!(f32)),
        derive(parse_quote!(f64)),
    ];

    quote!(
        impl#impl_generics #num_traits::ToPrimitive for #struct_ident#ty_generics
        #where_clause
        {
            #(#methods)*
        }
    )
    .into()
}

/// Derives [`Pow`]`<u8>`, [`Pow`]`<&'_ u8>` for `Self`, `&'_ Self`.
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`Pow`]: https://docs.rs/num-traits/0.2/num_traits/pow/trait.Pow.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(Pow_u8, attributes(modtype))]
pub fn pow_u8(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    pow(input, parse_quote!(u8))
}

/// Derives [`Pow`]`<u16>`, [`Pow`]`<&'_ u16>` for `Self`, `&'_ Self`.
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`Pow`]: https://docs.rs/num-traits/0.2/num_traits/pow/trait.Pow.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(Pow_u16, attributes(modtype))]
pub fn pow_u16(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    pow(input, parse_quote!(u16))
}

/// Derives [`Pow`]`<u32>`, [`Pow`]`<&'_ u32>` for `Self`, `&'_ Self`.
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`Pow`]: https://docs.rs/num-traits/0.2/num_traits/pow/trait.Pow.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(Pow_u32, attributes(modtype))]
pub fn pow_u32(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    pow(input, parse_quote!(u32))
}

/// Derives [`Pow`]`<usize>`, [`Pow`]`<&'_ usize>` for `Self`, `&'_ Self`.
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
///
/// [`Pow`]: https://docs.rs/num-traits/0.2/num_traits/pow/trait.Pow.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(Pow_usize, attributes(modtype))]
pub fn pow_usize(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    pow(input, parse_quote!(usize))
}

/// Derives [`Integer`].
///
/// # Requirements
///
/// - `Self: `[`From`]`<#InnerValue>`.
/// - `Self: `[`Copy`].
/// - `Self: `[`Zero`].
/// - `Self: `[`Ord`]. (required by [`Integer`] itself)
/// - `Self: `[`Num`]. (required by [`Integer`] itself)
///
/// [`Integer`]: https://docs.rs/num-integer/0.1/num_integer/trait.Integer.html
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
/// [`Copy`]: https://doc.rust-lang.org/nightly/core/marker/trait.Copy.html
/// [`Zero`]: https://docs.rs/num-traits/0.2/num_traits/identities/trait.Zero.html
/// [`Ord`]: https://doc.rust-lang.org/nightly/core/cmp/trait.Ord.html
/// [`Num`]: https://docs.rs/num-traits/0.2/num_traits/trait.Num.html
#[proc_macro_derive(Integer, attributes(modtype))]
pub fn integer(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        modulus,
        std,
        num_traits,
        num_integer,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #num_integer::Integer for #struct_ident#ty_generics
        #where_clause
        {
            #[inline]
            fn div_floor(&self, other: &Self) -> Self {
                fn static_assert_copy<T: #std::marker::Copy>() {}
                static_assert_copy::<Self>();

                *self / *other
            }

            #[inline]
            fn mod_floor(&self, other: &Self) -> Self {
                *self / *other
            }

            #[inline]
            fn gcd(&self, other: &Self) -> Self {
                let max = #std::cmp::max(self.#field_ident, other.#field_ident);
                <Self as #std::convert::From<#field_ty>>::from(max)
            }

            #[inline]
            fn lcm(&self, other: &Self) -> Self {
                let mut value = #num_integer::lcm(self.#field_ident, other.#field_ident);
                if value >= #modulus {
                    value %= #modulus;
                }
                <Self as #std::convert::From<#field_ty>>::from(value)
            }

            #[inline]
            fn divides(&self, other: &Self) -> bool {
                <Self as #num_integer::Integer>::is_multiple_of(self, other)
            }

            #[inline]
            fn is_multiple_of(&self, other: &Self) -> bool {
                <#field_ty as #num_traits::Zero>::is_zero(&(*self % *other))
            }

            #[inline]
            fn is_even(&self) -> bool {
                <#field_ty as #num_integer::Integer>::is_even(&self.#field_ident)
            }

            #[inline]
            fn is_odd(&self) -> bool {
                <#field_ty as #num_integer::Integer>::is_odd(&self.#field_ident)
            }

            #[inline]
            fn div_rem(&self, other: &Self) -> (Self, Self) {
                (*self / *other, <Self as #num_traits::Zero>::zero())
            }
        }
    )
    .into()
}

/// Derives [`ToBigUint`].
///
/// # Requirements
///
/// Nothing.
///
/// [`ToBigUint`]: https://docs.rs/num-bigint/0.2/num_bigint/biguint/trait.ToBigUint.html
#[proc_macro_derive(ToBigUint, attributes(modtype))]
pub fn to_big_uint(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    ref_unary_transparent(
        input,
        |Input { num_bigint, .. }| parse_quote!(#num_bigint::ToBigUint),
        parse_quote!(to_biguint),
        |Input {
             std, num_bigint, ..
         }| parse_quote!(#std::option::Option<#num_bigint::BigUint>),
    )
}

/// Derives [`ToBigInt`].
///
/// # Requirements
///
/// Nothing.
///
/// [`ToBigInt`]: https://docs.rs/num-bigint/0.2/num_bigint/biguint/trait.ToBigInt.html
#[proc_macro_derive(ToBigInt, attributes(modtype))]
pub fn to_big_int(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    ref_unary_transparent(
        input,
        |Input { num_bigint, .. }| parse_quote!(#num_bigint::ToBigInt),
        parse_quote!(to_bigint),
        |Input {
             std, num_bigint, ..
         }| parse_quote!(#std::option::Option<#num_bigint::BigInt>),
    )
}

/// Implement `Self::new: fn(#InnerValue) -> Self`.
///
/// # Requirements
///
/// - `Self: From<#InnerValue>`.
///
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(new, attributes(modtype))]
pub fn new(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct_method(input, |input| {
        let Input {
            std,
            struct_vis,
            struct_ident,
            field_ty,
            ..
        } = input;
        let doc = format!("Constructs a new `{}`.", struct_ident);
        parse_quote! {
            #[doc = #doc]
            #[inline]
            #struct_vis fn new(value: #field_ty) -> Self {
                <Self as #std::convert::From<#field_ty>>::from(value)
            }
        }
    })
}

/// Derives `Self::get: fn(Self) -> #InnerValue`.
///
/// # Requirements
///
/// Nothing.
///
/// [`From`]: https://doc.rust-lang.org/nightly/core/convert/trait.From.html
#[proc_macro_derive(get, attributes(modtype))]
pub fn get(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct_method(input, |input| {
        let Input {
            struct_vis,
            field_ident,
            field_ty,
            ..
        } = input;
        parse_quote! {
            #[doc = "Gets the inner value."]
            #[inline]
            #struct_vis fn get(self) -> #field_ty {
                self.#field_ident
            }
        }
    })
}

fn struct_method(
    input: proc_macro::TokenStream,
    item_fn: fn(&Input) -> ItemFn,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        struct_ident,
        generics,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let item_fn = item_fn(&input);
    quote!(
        impl#impl_generics #struct_ident#ty_generics
        #where_clause
        {
            #item_fn
        }
    )
    .into()
}

fn fmt(input: proc_macro::TokenStream, trait_ident: Ident) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    quote!(
        impl#impl_generics #std::fmt::#trait_ident for #struct_ident#ty_generics
        #where_clause
        {
            #[inline]
            fn fmt(&self, fmt: &mut #std::fmt::Formatter) -> #std::fmt::Result {
                <#field_ty as #std::fmt::#trait_ident>::fmt(&self.#field_ident, fmt)
            }
        }
    )
    .into()
}

fn bin_almost_transparent(
    input: proc_macro::TokenStream,
    trait_ident: Ident,
    fn_ident: Ident,
    op: fn(&Expr, &Expr, &Expr) -> Expr,
) -> proc_macro::TokenStream {
    bin(input, trait_ident, |input, rhs_ty| {
        let Input {
            modulus,
            struct_ident,
            generics,
            field_ident,
            ..
        } = input;
        let (_, ty_generics, _) = generics.split_for_impl();

        let expr = op(
            &parse_quote!(self.#field_ident),
            &parse_quote!(rhs.#field_ident),
            &modulus,
        );
        let struct_expr = input.construct_self(false, None);
        parse_quote! {
            #[inline]
            fn #fn_ident(self, rhs: #rhs_ty) -> #struct_ident#ty_generics {
                let mut #field_ident = #expr;
                if #field_ident >= #modulus {
                    #field_ident %= #modulus;
                }
                #struct_expr
            }
        }
    })
}

fn bin(
    input: proc_macro::TokenStream,
    trait_ident: Ident,
    derive_fn: impl Fn(&Input, &Type) -> ItemFn,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        std,
        no_impl_for_ref,
        struct_ident,
        generics,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let derive = |impl_generics: &ImplGenerics, lhs_ty: Type, rhs_ty: Type| -> _ {
        let item_fn = derive_fn(&input, &rhs_ty);
        quote! {
            impl#impl_generics #std::ops::#trait_ident<#rhs_ty> for #lhs_ty
            #where_clause
            {
                type Output = #struct_ident#ty_generics;

                #item_fn
            }
        }
    };

    let mut ret = derive(
        &impl_generics,
        parse_quote!(#struct_ident#ty_generics),
        parse_quote!(#struct_ident#ty_generics),
    );

    if !no_impl_for_ref {
        ret.extend(derive(
            &impl_generics,
            parse_quote!(&'_ #struct_ident#ty_generics),
            parse_quote!(#struct_ident#ty_generics),
        ));

        ret.extend(derive(
            &impl_generics,
            parse_quote!(#struct_ident#ty_generics),
            parse_quote!(&'_ #struct_ident#ty_generics),
        ));

        ret.extend(derive(
            &impl_generics,
            parse_quote!(&'_ #struct_ident#ty_generics),
            parse_quote!(&'_ #struct_ident#ty_generics),
        ));
    }

    ret.into()
}

fn bin_assign(
    input: proc_macro::TokenStream,
    trait_ident: Ident,
    fn_ident: Ident,
    bin_op: BinOp,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        std,
        no_impl_for_ref,
        struct_ident,
        generics,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let derive = |rhs_ty: Type, rhs_deref: bool| -> _ {
        let star_token = if rhs_deref { quote!(*) } else { quote!() };
        quote! {
            impl#impl_generics #std::ops::#trait_ident<#rhs_ty> for #struct_ident#ty_generics
            #where_clause
            {
                #[inline]
                fn #fn_ident(&mut self, rhs: #rhs_ty) {
                    fn static_assert_copy<T: #std::marker::Copy>() {}
                    static_assert_copy::<Self>();
                    *self = *self #bin_op #star_token rhs;
                }
            }
        }
    };

    let mut ret = derive(parse_quote!(Self), false);
    if !no_impl_for_ref {
        ret.extend(derive(parse_quote!(&'_ Self), true));
    }
    ret.into()
}

fn identity(
    input: proc_macro::TokenStream,
    trait_ident: Ident,
    value: Ident,
    is: Ident,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        num_traits,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let struct_expr = input.construct_self(
        true,
        Some(parse_quote!(<#field_ty as #num_traits::#trait_ident>::#value())),
    );

    quote!(
        impl#impl_generics #num_traits::#trait_ident for #struct_ident#ty_generics
        #where_clause
        {
            #[inline]
            fn #value() -> Self {
                #struct_expr
            }

            #[inline]
            fn #is(&self) -> bool {
                <#field_ty as #num_traits::#trait_ident>::#is(&self.#field_ident)
            }
        }
    )
    .into()
}

fn checked_bin(
    input: proc_macro::TokenStream,
    trait_ident: Ident,
    fn_ident: Ident,
    return_none_if_rhs_is_zero: bool,
    op: BinOp,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        num_traits,
        struct_ident,
        generics,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let expr: Expr = if return_none_if_rhs_is_zero {
        parse_quote! {
            if <Self as #num_traits::Zero>::is_zero(rhs) {
                None
            } else {
                Some(*self #op *rhs)
            }
        }
    } else {
        parse_quote!(Some(*self #op *rhs))
    };

    quote!(
        impl#impl_generics #num_traits::#trait_ident for #struct_ident#ty_generics
            #where_clause
        {
            #[inline]
            fn #fn_ident(&self, rhs: &Self) -> #std::option::Option<Self> {
                fn static_assert_copy<T: #std::marker::Copy>() {}
                static_assert_copy::<Self>();
                #expr
            }
        }
    )
    .into()
}

fn pow(input: proc_macro::TokenStream, rhs_ty: Type) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let Input {
        std,
        num_traits,
        no_impl_for_ref,
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = try_syn!(Input::try_from(input));
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let derive = |lhs_ty: &Type, rhs_ty: &Type| -> _ {
        quote! {
            impl#impl_generics #num_traits::Pow<#rhs_ty> for #lhs_ty
                #where_clause
            {
                type Output = #struct_ident#ty_generics;

                #[inline]
                fn pow(self, rhs: #rhs_ty) -> #struct_ident#ty_generics {
                    let value = <#field_ty as #num_traits::Pow<#rhs_ty>>::pow(self.#field_ident, rhs);
                    <#struct_ident#ty_generics as #std::convert::From<#field_ty>>::from(value)
                }
            }
        }
    };

    let mut ret = derive(&parse_quote!(#struct_ident#ty_generics), &rhs_ty);
    if !no_impl_for_ref {
        ret.extend(derive(
            &parse_quote!(#struct_ident#ty_generics),
            &parse_quote!(&'_ #rhs_ty),
        ));
        ret.extend(derive(
            &parse_quote!(&'_ #struct_ident#ty_generics),
            &rhs_ty,
        ));
        ret.extend(derive(
            &parse_quote!(&'_ #struct_ident#ty_generics),
            &parse_quote!(&'_ #rhs_ty),
        ));
    }
    ret.into()
}

fn ref_unary_transparent(
    input: proc_macro::TokenStream,
    trait_ty: fn(&Input) -> Type,
    fn_ident: Ident,
    output_ty: fn(&Input) -> Type,
) -> proc_macro::TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let input = try_syn!(Input::try_from(input));
    let Input {
        struct_ident,
        generics,
        field_ident,
        field_ty,
        ..
    } = &input;
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let trait_ty = trait_ty(&input);
    let output_ty = output_ty(&input);
    quote!(
        impl#impl_generics #trait_ty for #struct_ident#ty_generics
        #where_clause
        {
            #[inline]
            fn #fn_ident(&self) -> #output_ty {
                <#field_ty as #trait_ty>::#fn_ident(&self.#field_ident)
            }
        }
    )
    .into()
}

struct Input {
    modulus: Expr,
    std: Path,
    num_traits: Path,
    num_integer: Path,
    num_bigint: Path,
    no_impl_for_ref: bool,
    struct_vis: Visibility,
    struct_ident: Ident,
    generics: Generics,
    field_ident: Ident,
    field_ty: Type,
    other_fields: Vec<(Ident, Type)>,
}

impl Input {
    fn construct_self(&self, path_is_self: bool, value_expr: Option<Expr>) -> ExprStruct {
        let Self {
            std,
            struct_ident,
            field_ident,
            other_fields,
            ..
        } = self;

        let struct_ident_or_self: Path = if path_is_self {
            parse_quote!(Self)
        } else {
            parse_quote!(#struct_ident)
        };

        let value_field = match value_expr {
            None => quote!(#field_ident),
            Some(value_expr) => quote!(#field_ident: #value_expr),
        };

        let assign = other_fields
            .iter()
            .map(|(ident, ty)| quote!(#ident: <#ty as #std::default::Default>::default()));

        parse_quote! {
            #struct_ident_or_self {
                #value_field,
                #(#assign,)*
            }
        }
    }
}

impl TryFrom<DeriveInput> for Input {
    type Error = syn::Error;

    fn try_from(input: DeriveInput) -> syn::Result<Self> {
        static TARGET_ATTR: &'static str = "modtype";

        fn error_on_target_attr(meta: &Meta) -> syn::Result<()> {
            match find_target_attr(meta) {
                None => Ok(()),
                Some(span) => Err(syn::Error::new(
                    span,
                    format!("`{}` not allowed here", TARGET_ATTR),
                )),
            }
        }

        fn find_target_attr(meta: &Meta) -> Option<Span> {
            match meta {
                Meta::Word(ident)
                | Meta::NameValue(MetaNameValue { ident, .. })
                | Meta::List(MetaList { ident, .. })
                    if ident == TARGET_ATTR =>
                {
                    Some(ident.span())
                }
                Meta::Word(_) | Meta::NameValue(_) => None,
                Meta::List(list) => list
                    .nested
                    .iter()
                    .flat_map(|m| match m {
                        NestedMeta::Meta(m) => find_target_attr(m),
                        NestedMeta::Literal(Lit::Str(s)) if s.value() == TARGET_ATTR => {
                            Some(s.span())
                        }
                        NestedMeta::Literal(_) => None,
                    })
                    .next(),
            }
        }

        fn put_expr_for(lhs: Span, rhs: &Lit, dist: &mut Option<Expr>) -> syn::Result<()> {
            let expr = match rhs {
                Lit::Int(int) => Ok(parse_quote!(#int)),
                Lit::Str(s) => s.parse(),
                rhs => Err(rhs.to_error("expected string or unsigned 64-bit integer")),
            }?;
            match mem::replace(dist, Some(expr)) {
                Some(_) => Err(syn::Error::new(lhs, "multiple definitions")),
                None => Ok(()),
            }
        }

        fn put_path_for(lhs: Span, rhs: &Lit, dist: &mut Option<Path>) -> syn::Result<()> {
            let path = match rhs {
                Lit::Str(s) => s.parse::<Path>(),
                rhs => Err(rhs.to_error("expected string literal")),
            }?;
            match mem::replace(dist, Some(path)) {
                Some(_) => Err(syn::Error::new(lhs, "multiple definitions")),
                None => Ok(()),
            }
        }

        fn put_true_for(word: Span, dist: &mut bool) -> syn::Result<()> {
            if mem::replace(dist, true) {
                Err(syn::Error::new(word, "multiple definitions"))
            } else {
                Ok(())
            }
        }

        trait SpannedExt {
            fn to_error(&self, mes: impl fmt::Display) -> syn::Error;
        }

        impl<T: Spanned> SpannedExt for T {
            fn to_error(&self, mes: impl fmt::Display) -> syn::Error {
                syn::Error::new(self.span(), mes)
            }
        }

        let DeriveInput {
            attrs,
            vis: struct_vis,
            ident: struct_ident,
            generics,
            data,
        } = input;

        let mut modulus = None;
        let mut std = None;
        let mut num_traits = None;
        let mut num_integer = None;
        let mut num_bigint = None;
        let mut no_impl_for_ref = false;

        let mut put_expr_or_path = |name_value: &MetaNameValue| -> syn::Result<_> {
            let span = name_value.span();
            let MetaNameValue { ident, lit, .. } = name_value;
            if ident == "modulus" {
                put_expr_for(ident.span(), lit, &mut modulus)
            } else if ident == "std" {
                put_path_for(ident.span(), lit, &mut std)
            } else if ident == "num_traits" {
                put_path_for(ident.span(), lit, &mut num_traits)
            } else if ident == "num_integer" {
                put_path_for(ident.span(), lit, &mut num_integer)
            } else if ident == "num_bigint" {
                put_path_for(ident.span(), lit, &mut num_bigint)
            } else if ident == "no_impl_for_ref" {
                Err(syn::Error::new(span, "expected `no_impl_for_ref`"))
            } else {
                Err(syn::Error::new(span, "unknown identifier"))
            }
        };

        let mut put_true = |word: &Ident| -> syn::Result<_> {
            if ["modulus", "std", "num_traits", "num_integer", "num_bigint"]
                .contains(&word.to_string().as_str())
            {
                Err(word.to_error(format!("expected `{} = #LitStr`", word)))
            } else if word == "no_impl_for_ref" {
                put_true_for(word.span(), &mut no_impl_for_ref)
            } else {
                Err(word.to_error("unknown identifier"))
            }
        };

        attrs
            .iter()
            .flat_map(Attribute::parse_meta)
            .try_for_each::<_, syn::Result<_>>(|meta| {
                if_chain! {
                    if let Meta::List(MetaList { ident, nested, .. }) = &meta;
                    if ident == TARGET_ATTR;
                    then {
                        for nested in nested {
                            match nested {
                                NestedMeta::Meta(Meta::Word(word)) => put_true(word)?,
                                NestedMeta::Meta(Meta::NameValue(kv)) => put_expr_or_path(kv)?,
                                _ => return Err(nested.to_error("expected `#Ident` or `#Ident = #Lit`")),
                            }
                        }
                        Ok(())
                    } else {
                        error_on_target_attr(&meta)
                    }
                }
            })?;

        let modulus = modulus.ok_or_else(|| struct_ident.to_error("`modulus` required"))?;
        let std = std.unwrap_or_else(|| parse_quote!(::std));
        let num_traits = num_traits.unwrap_or_else(|| parse_quote!(::num::traits));
        let num_integer = num_integer.unwrap_or_else(|| parse_quote!(::num::integer));
        let num_bigint = num_bigint.unwrap_or_else(|| parse_quote!(::num::bigint));

        let fields = match data {
            Data::Struct(DataStruct { fields, .. }) => Ok(fields),
            Data::Enum(DataEnum { enum_token, .. }) => {
                Err(enum_token.to_error("expected a struct"))
            }
            Data::Union(DataUnion { union_token, .. }) => {
                Err(union_token.to_error("expected a struct"))
            }
        }?;

        let named = match fields {
            Fields::Named(FieldsNamed { named, .. }) => Ok(named),
            fields => Err(fields.to_error("expected named fields")),
        }?;
        let named_span = named.span();

        let (mut value_field, mut other_fields) = (None, vec![]);
        'l: for field in named {
            for attr in &field.attrs {
                if let Ok(meta) = attr.parse_meta() {
                    if_chain! {
                        if let Meta::List(MetaList { ident, nested, .. }) = &meta;
                        if ident == TARGET_ATTR;
                        then {
                            if ![parse_quote!(value), parse_quote!(value,)].contains(nested) {
                                return Err(nested.to_error("expected `value` or `value,`"));
                            }
                            value_field = Some(field);
                            continue 'l;
                        } else {
                            error_on_target_attr(&meta)?;
                        }
                    }
                }
            }
            other_fields.push((field.ident.unwrap(), field.ty));
        }

        let Field {
            vis,
            ident,
            ty: field_ty,
            ..
        } = value_field.ok_or_else(|| {
            syn::Error::new(named_span, format!("`#[{}(value)]` not found", TARGET_ATTR))
        })?;
        let field_ident = ident.unwrap();

        if vis != Visibility::Inherited {
            return Err(vis.to_error("the field visibility must be `Inherited`"));
        }

        if !field_ident.to_string().starts_with("__") {
            return Err(field_ident.to_error("the field name must start with \"__\""));
        }

        Ok(Self {
            modulus,
            std,
            num_traits,
            num_integer,
            num_bigint,
            no_impl_for_ref,
            struct_vis,
            struct_ident,
            generics,
            field_ident,
            field_ty,
            other_fields,
        })
    }
}
