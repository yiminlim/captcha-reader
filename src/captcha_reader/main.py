from captcha import Captcha

def main():
    captcha_reader = Captcha()
    input_filepath = input("Please enter the input filepath: ")
    captcha_reader(input_filepath, "result/output.txt")
    print("Captcha reading completed. Output saved to 'result' folder as output.txt")
if __name__ == "__main__":
    main()