import random
import string

def change_my_name(length):
    """
    This function returns a random string of length length.
    """
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    return result_str

