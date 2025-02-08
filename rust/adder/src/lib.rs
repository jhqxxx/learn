/*
 * @Author: jhq
 * @Date: 2024-03-03 15:28:19
 * @LastEditTime: 2024-03-03 15:31:28
 * @Description: 
 */
pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn another() {
        panic!("Make this test fail");
    }
}
