
@torch.no_grad()
def main():
    x = torch.randn(10,1)
    a = nn.Parameter(torch.randn(1,10), requires_grad=True)
    y = torch.mm(a,x)
    s=2*y
    s.backward()
    w=2
main()