use std::error;
/*
 * @Author: jhq
 * @Date: 2024-03-10 17:43:34
 * @LastEditTime: 2024-03-21 22:34:38
 * @Description:
 */
use std::env;
use std::error::Error;
use std::fs;

pub fn run(config: Config) -> Result<(), Box<dyn Error>> {
    let contents = fs::read_to_string(config.filename)?;
    let result = if config.case_sensitive {
        search(&config.query, &contents)
    } else {
        search_case_insensitive(&config.query, &contents)
    };

    println!("With text:\n {}", contents);
    for line in result {
        println!("{}", line);
    }
    Ok(())
}
pub struct Config {
    pub query: String,
    pub filename: String,
    pub case_sensitive: bool,
}
impl Config {
    pub fn new(mut args: std::env::Args) -> Result<Config, &'static str> {
        if args.len() < 3 {
            return Err("not enough arguments");
        }
        args.next();
        let query = match args.next() {
            Some(arg) => arg,
            None => return Err("Don't get a query string"),
        };
        let filename = match args.next() {
            Some(arg) => arg,
            None => return Err("Don't get a file string"),
        };
        let case_sensitive = env::var("CASE_INSENSITIVE")
                                    .unwrap_or(String::from("")) == "1";
        // let case_sensitive = match env::var("CASE_INSENSITIVE") {
        //     Ok(flag) => {
        //         &flag == "1"
        //     },
        //     Err(err) => false,
        // };

        println!("case_sensitive: {}", case_sensitive);
        Ok(Config {
            query,
            filename,
            case_sensitive,
        })
    }
}

pub fn search<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    println!("kkkkkkkkkkkkkk");
    contents
        .lines()
        .filter(|line| line.contains(query))
        .collect()
    // let mut results = Vec::new();
    // for line in contents.lines() {
    //     if line.contains(query) {
    //         results.push(line);
    //     }
    // }
    // results
}

pub fn search_case_insensitive<'a>(query: &str, contents: &'a str) -> Vec<&'a str> {
    contents
        .lines()
        .filter(|line| line.to_lowercase().contains(query))
        .collect()
    // let mut results = Vec::new();
    // let query = query.to_lowercase();

    // for line in contents.lines() {
    //     if line.to_lowercase().contains(&query) {
    //         results.push(line);
    //     }
    // }
    // results
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn case_sensitive() {
        let query = "duct";
        let contents = "\
Rust:
safe, fast, productive.
Pike three.
Duct tape.";
        assert_eq!(vec!["safe, fast, productive."], search(query, contents))
    }

    #[test]
    fn case_insensitive() {
        let query = "rUsT";
        let contents = "\
Rust:
safe, fast, productive. 
Pike three. 
Trust me.";
        assert_eq!(
            vec!["Rust:", "Trust me."],
            search_case_insensitive(query, contents)
        )
    }
}
