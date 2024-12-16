use proc_macro::TokenStream;

fn impl_msg_pack_format_trait(ast: syn::DeriveInput) -> TokenStream {
    let ident = ast.ident;

    quote::quote! {
        impl MSGPackFormat for #ident {
            fn to_msgpack(&self) -> NNResult<Vec<u8>> {
                Ok(rmp_serde::to_vec(&self)?)
            }

            fn from_msgpack(buff: &[u8]) -> NNResult<Box<Self>>
            where
                Self: Sized,
            {
                Ok(Box::new(rmp_serde::from_slice::<Self>(buff)?))
            }
        }
    }
    .into()
}

fn impl_layer_trait(ast: syn::DeriveInput) -> TokenStream {
    let ident = ast.ident;

    quote::quote! {
        impl Layer for #ident {
            fn layer_type(&self) -> &str {
               stringify!(#ident)
            }

            fn as_any(&self) -> &dyn std::any::Any {
                self
            }
        }

        impl MSGPackFormat for #ident {
            fn to_msgpack(&self) -> NNResult<Vec<u8>> {
                Ok(rmp_serde::to_vec(&self)?)
            }

            fn from_msgpack(buff: &[u8]) -> NNResult<Box<Self>>
            where
                Self: Sized,
            {
                Ok(Box::new(rmp_serde::from_slice::<Self>(buff)?))
            }
        }
    }
    .into()
}

fn impl_activation_trait(ast: syn::DeriveInput) -> TokenStream {
    let ident = ast.ident;

    quote::quote! {
        impl ActivationFunction for #ident {
            fn activation(&self) -> &str {
                stringify!(#ident)
            }

            fn from_activation(_activation: &str) -> NNResult<Box<dyn ActivationFunction>>
            where
                Self: Sized,
            {
                Ok(Box::new(#ident))
            }
        }
    }
    .into()
}

fn impl_cost_trait(ast: syn::DeriveInput) -> TokenStream {
    let ident = ast.ident;

    quote::quote! {
        impl CostFunction for #ident {
            fn cost_name(&self) -> &str {
                stringify!(#ident)
            }

            fn from_cost(_cost: &str) -> NNResult<Box<dyn CostFunction>>
            where
                Self: Sized,
            {
                Ok(Box::new(#ident))
            }
        }
    }
    .into()
}

#[proc_macro_derive(MSGPackFormat)]
pub fn msg_pack_format_derive_macro(item: TokenStream) -> TokenStream {
    let ast = syn::parse(item).unwrap();

    impl_msg_pack_format_trait(ast)
}

#[proc_macro_derive(Layer)]
pub fn layer_derive_macro(item: TokenStream) -> TokenStream {
    let ast = syn::parse(item).unwrap();

    impl_layer_trait(ast)
}

#[proc_macro_derive(ActivationFunction)]
pub fn activation_derive_macro(item: TokenStream) -> TokenStream {
    let ast = syn::parse(item).unwrap();

    impl_activation_trait(ast)
}

#[proc_macro_derive(CostFunction)]
pub fn cost_derive_macro(item: TokenStream) -> TokenStream {
    let ast = syn::parse(item).unwrap();

    impl_cost_trait(ast)
}
