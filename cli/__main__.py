import os
import sys

def main():
    server = os.environ.get('FRIDAY_SERVER')
    print("FRIDAY cli using server:")
    print(server)
main()