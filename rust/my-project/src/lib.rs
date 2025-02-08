/*
 * @Author: jhq
 * @Date: 2024-02-17 17:35:10
 * @LastEditTime: 2024-02-17 18:27:57
 * @Description: 
 */
mod front_of_house; 

fn serve_order() {}

mod back_of_house {
    fn fix_incorrect_order() {
        cook_order();
        super::serve_order();
    }
    fn cook_order() {}

    pub struct Breakfast {
        pub toast: String,
        seasonal_fruit: String,
    }

    pub enum Appetizer {  // 公共枚举选项默认是公共的
        Soup,
        Salad,
    }

    impl Breakfast {
        pub fn summer(toast: &str) -> Breakfast {
            Breakfast {
                toast: String::from(toast),
                seasonal_fruit: String::from("peaches"),
            }
        }
    }
}

pub use crate::front_of_house::hosting;

pub fn eat_at_restaurant() {

    let mut meal = back_of_house::Breakfast::summer("Rye");
    meal.toast = String::from("Wheat");
    

    crate::front_of_house::hosting::add_to_waitlist();   // 绝对路径
    front_of_house::hosting::add_to_waitlist();   // 相对路径
    hosting::add_to_waitlist();
}