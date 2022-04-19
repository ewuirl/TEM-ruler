import argparse

def simple_test_func():
    print("potato")

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Test running from the command line")
    parser.add_argument("print_string", type=str, help="thing to print")
    
    args = parser.parse_args()
    print_string = args.print_string
    print(print_string)
    print("saving a file")
    with open(f"{print_string}.txt", "w") as file:
        file.write(print_string)

if __name__ == "__main__":
    main()
#     # Argument parser
#     parser = argparse.ArgumentParser(description="Test running from the command line")
#     parser.add_argument("print_string", type=str, help="thing to print")
    
#     args = parser.parse_args()
#     print_string = args.print_string
#     print(print_string)
#     print("saving a file")
#     with open(f"{print_string}.txt", "w") as file:
#         file.write(print_string)
