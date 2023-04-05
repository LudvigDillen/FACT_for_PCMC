import torch


def main():
    x = torch.rand(5, 3)
    print(x)
    print(torch.cuda.is_available())

if __name__ == "__main__":
    main()