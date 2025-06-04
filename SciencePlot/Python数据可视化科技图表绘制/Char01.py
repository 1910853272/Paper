#【例1-1】在控制台进行命令的输入示例。依次输入以下代码，并观察输出结果。

# >> > 3 + 8  # 在提示符后输入命令，按Enter键
# 11
# >> > print('Hello Ding')
# Hello
# Ding
# >> > a = 1
# >> > b = 2
# >> > print(a + b)
# 3
# >> > c = 4;
# d = 2
# >> > print(c * d)
# 8
# >> > exit()  # 输入该命令，按Enter键即可退出交互式编程环境


#【例1-2】注释示例。依次输入以下代码，并观察输出结果。
# 这是单行注释
print("This is a comment!")  # 执行print语句，输出：This is a comment!

# 这是多行注释的起始处
'''
a=1 
b=4 
print("These are two comments!")			# 注释部分，不参与运行
'''
 # 这是多行注释的结尾处
print("These are multiple comments!")  # 输出：These are multiple comments!

#【例1-3】注释示例。依次输入以下代码，并观察输出结果。



item_01_ding = 2
item_02_zhang = 6
item_03_liu = 4
# 使用多行连接符
total1 = item_01_ding + item_02_zhang + item_03_liu
total2 = item_01_ding + \
        item_02_zhang + \
        item_03_liu
print(total1)  # 输出：12
print(total2)




days = ['Monday', 'Tuesday', 'Wednesday',  # 不需要使用多行连接符
        'Thursday', 'Friday']
print(days)  # 输出：['Monday','Tuesday','Wednesday','Thursday','Friday']

#【例1-4】输入输出示例。依次输入以下代码，并观察输出结果。
# 输出字符串
print("Hello,world!")  # 输出：Hello,world!

# 输出变量
x = 10
y = 20
print("The value of x is:", x)  # 输出：The value of x is:10
print("The value of y is:", y)  # 输出：The value of y is:20



# 格式化输出
name = "Alice"
age = 30
fname = 'Tom'
print("My name is {} and I am {} years old.my friend is {} ".format(name, age,fname))
# 输出：My name is Alice and I am 30 years old.



# 接收用户输入
name = input("Please enter your name:")  # 按提示输入：Ding
print("Hello,", name)  # 输出：Hello,ding

# 将输入的字符串转换为整数
age = int(input("Please enter your age:"))  # 按提示输入：12
print("Your age is:", age)  # 输出：Your age is: 12

# 连续输入多个值并以空格分隔
values = input("Please enter multiple values separated by space:")
# 按提示输入：Ding Zhang  Liu
values_list = values.split()  # 将输入的字符串分割为列表
print("You entered:", values_list)
# 输出：You entered: ['Ding','Zhang','Liu']

# 连续输入多个值并以逗号分隔
values = input("Please enter multiple values separated by comma:")
# 按提示输入：Bin,Zhi,Lan
values_list = values.split(",")  # 将输入的字符串分割为列表
print("You entered:", values_list)
# 输出：You entered: ['Bin','Zhi','Lan']



#【例1-5】运算符应用示例。依次输入以下代码，并观察输出结果。
# 算术运算符：用于执行基本的数学运算，如加法、减法、乘法等
a = 10
b = 5
print("Addition:", a + b)  # 加法，输出：Addition：15
print("Subtraction:", a - b)  # 减法，输出：Subtraction：5
print("Multiplication:", a * b)  # 乘法，输出：Multiplication：50
print("Division:", a / b)  # 除法，输出：Division：2.0
print("Modulus:", a % b)  # 取余，输出：Modulus：0
print("Exponentiation:", a ** b)  # 幂运算，输出：Exponentiation：100000
print("Floor Division:", a // b)  # 地板除法，输出：Floor Division：2

# 比较运算符：用于比较两个值之间的关系，返回布尔值（True或False）
print("a>b:", a > b)  # 输出：a>b:True
print("a<b:", a < b)  # 输出：a<b:False
print("a==b:", a == b)  # 输出：a==b:False
print("a!=b:", a != b)  # 输出：a!=b:True
print("a>=b:", a >= b)  # 输出：a>=b:True
print("a<=b:", a <= b)  # 输出：a<=b:False




# 赋值运算符：用于给变量赋值
x = 5
y = 10
x += y  # 等价于x=x+y
print("x:", x)  # 输出：x:15
x -= y  # 等价于x=x-y
print("x:", x)  # 输出：x:5
x *= y  # 等价于x=x*y
print("x:", x)  # 输出：x:50

x /= y  # 等价于x=x/y
print("x:", x)  # 输出：x:5.0
x %= y  # 等价于x=x%y
print("x:", x)  # 输出：x:5.0

x **= y  # 等价于x=x**y
print("x:", x)  # 输出：x:9765625.0
x //= y  # 等价于x=x//y
print("x:", x)  # 输出：x:976562.0




# 逻辑运算符：用于在多个条件之间进行逻辑运算，返回布尔值
x = True  # 1
y = False  # 0
print("x and y:", x and y)  # 输出：x and y:False
print("x or y:", x or y)  # 输出：x or y:True
print("not x:", not x)  # 输出：not x:False



# 位运算符：用于对整数进行位操作
a = 60     
b = 13
print("Bitwise AND:", a & b)  # 按位与，输出：Bitwise AND:12
print("Bitwise OR:", a | b)  # 按位或，输出：Bitwise OR 61
print("Bitwise XOR:", a ^ b)  # 按位异或，输出：Bitwise XOR:49
print("Bitwise NOT:", ~a)  # 按位取反，输出：Bitwise NOT:-61
print("Bitwise Left Shift:", a << 2)  # 左移2位：Bitwise Left Shift:240
print("Bitwise Right Shift:", a >> 2)  # 右移2位：Bitwise Right Shift:15



# 成员运算符：用于检查一个值是否存在于序列中。
my_list = [1, 2, 3, 4, 5]
print("Is 3 in my_list:", 3 in my_list)  # 输出：Is 3 in my_list:True
print("Is 6 not in my_list:",
      6 not in my_list)  # 输出：Is 6 not in my_list:True


# 身份运算符：用于检查两个对象是否引用同一个内存地址。
x = 5
y = 5
list1_5 = [1,2,8]
print("x is y:", x is list1_5)  # 输出：x is y:True
print("x is not y:", x is not y)  # 输出：x is not y:False



#【例1-6】列表示例。依次输入以下代码，并观察输出结果。
my_list = [1, 2, 3, 4, 5]  # 创建一个列表

print(my_list)
print(my_list[0])  # 访问列表元素，输出：1
print(my_list[2])  # 访问列表元素，输出：3
print(my_list[1:])  # 切片操作，输出：[2,3]

my_list.append(6)  # 添加元素
print(my_list)  # 输出：[1,2,3,4,5,6]
my_list[0] = 0  # 修改元素
print(my_list)  # 输出：[0,2,3,4,5,6]
del my_list[1]  # 删除元素
print(my_list)  # 输出：[0,3,4,5,6]




#【例1-7】元组示例。依次输入以下代码，并观察输出结果。
my_tuple = (1, 2, 3, 4, 5)  # 创建一个元组

print(my_tuple[0])  # 访问元组元素，输出：1
print(my_tuple[2])  # 访问元组元素，输出：3





#【例1-8】集合示例。依次输入以下代码，并观察输出结果。
my_set = {1, 2, 3, 4, 5}  # 创建一个集合
my_set.add(6)  # 添加元素
print(my_set)  # 输出：{1,2,3,4,5,6}
my_set.remove(3)  # 删除元素
print(my_set)  # 输出：{1,2,4,5,6}




# key-value
#【例1-9】字典示例。依次输入以下代码，并观察输出结果。
my_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}  # 创建一个字典
print(my_dict['name'])  # 访问字典元素，输出：'Alice'
print(my_dict['age'])  # 访问字典元素，输出：30

my_dict['age'] = 31  # 修改字典元素
print(my_dict)
# 输出：{'name':'Alice','age':31,'city':'New York'}
my_dict['gender'] = 'Female'  # 添加新元素
print(my_dict)
# 输出：{'name':'Alice','age':31,'city':'New York','gender':'Female'}
del my_dict['city']  # 删除元素
print(my_dict)
# 输出：{'name':'Alice','age':31,'gender':'Female'}




#【例1-10】字符串示例。依次输入以下代码，并观察输出结果。
my_string = "Hello,world!"  # 创建字符串

print(my_string[0])  # 访问字符串中的字符，输出：H
print(my_string[7])  # 访问字符串中的字符，输出：w
print(my_string[2:5])  # 切片操作，输出：llo
print(len(my_string))  # 字符串长度，输出：13

new_string = my_string + " How are you?"  # 字符串连接
print(new_string)  # 输出：Hello,world! How are you?

# 字符串格式化
name = "Alice"
age = 30
formatted_string = "My name is {} and I am {} years old.".format(name, age)
print(formatted_string)  # 输出：My name is Alice and I am 30 years old.

print(my_string.find("world"))  # 字符串查找，输出：7
new_string = my_string.replace("world", "Python")  # 字符串替换
print(new_string)  # 输出：Hello,Python!

# 字符串转换为大写或小写
print(my_string.upper())  # 输出：HELLO,WORLD!
print(my_string.lower())  # 输出：hello,world!

words = my_string.split(",")  # 字符串分割
print(words)  # 输出：['Hello','world!']

my_string_with_spaces = "  Hello,world!  "  # 字符串去除空格
print(my_string_with_spaces.strip())  # 输出：Hello,world!







#【例1-11】列表应用示例。依次输入以下代码，并观察输出结果。
my_list = [1, 2, 3, 4, 5]
print(my_list[0])  # 输出：1
print(my_list[-4])  # 输出：5

my_list = [1, 2, 3, 4, 5]
print(my_list[1:4])  # 输出：[2,3,4]
print(my_list[::1])  # 输出：[1,3,5]
# [x:y:z]
my_list = [1, 2, 3, 4, 5]
print(len(my_list))  # 输出：5

my_list = [1, 2, 3]
my_list.append(4)
print(my_list)  # 输出：[1,2,3,4]

list1 = [1, 2, 3]
list2 = ['a', 'b', 'c']

new_list = list1 + list2
print(new_list)  # 输出：[1,2,3,'a','b','c']

my_list = [1, 2]
repeated_list = my_list * 3
print(repeated_list)  # 输出：[1,2,1,2,1,2]

my_list = [1, 2, 3, 4, 5]
index = my_list.index(1)
print(index)  # 输出：2

my_list = [1, 2, 2, 3, 3, 3]
count = my_list.count(3)
print(count)  # 输出：2

my_list = [1, 2, 3, 4, 5]
del my_list[0]
print(my_list)  # 输出：[2,3,4,5]

my_list.remove(3)
print(my_list)  # 输出：[2,4,5]

my_list = [3, 1, 4, 1, 5, 9, 2]
my_list.sort
new_list_test = sorted(my_list,reversed)
print(my_list)  # 输出：[1,1,2,3,4,5,9]







#【例1-12】条件语句示例。依次输入以下代码，并观察输出结果。
x = 10
if x > 5:
    print("x大于5")  # 输出：x大于5

x = 3
if x > 5:
    print("x大于5")
else:
    print("x不大于5")
# 输出：x不大于5


x = 3
if x > 5:
    print("x大于5")
elif x == 5:
    print("x等于5")
else:
    print("x小于5")
# 输出：x小于5


x = 10
if x > 5:
    print("x大于5")
    if x == 10:
        print("x等于10")
else:
    print("x不大于5")
# 输出：x大于5
#       x等于10


x = 10
if x > 5: print("x大于5")  # 输出：x大于5






# 【例1-13】for循环语句示例。依次输入以下代码，并观察输出结果。
fruits = ["apple", "banana", "cherry"]
for fruito in fruits:
    print(fruito)

for char in "Python":
    print(char)

for i in range(5):  
    print(i)

adj = ["red", "big", "tasty"]
fruits = ["apple", "banana", "cherry"]

for ad in adj:
    for fruit in fruits:
        print(ad, fruit)

person = {"name": "Alice", "age": 30, "city": "New York"}

for key, value in person.items():
    print(key + ":", value)

for x in range(10):
    if x == 3:
        continue
    if x == 5:
        break
    print(x)






#【例1-14】while循环语句示例。依次输入以下代码，并观察输出结果。
x = 0
while x < 5:  # 打印出0到4的数字，在x小于5的条件下，循环会一直执行
    print(x)
    x += 1

password = ""
while password != "secret":  # 不断询问用户输入密码，直到输入的密码正确为止
    password = input("请输入密码：")
print("密码正确，登录成功！")

a, b = 0, 1
while a < 100:  # 打印出斐波那契数列中小于100的所有数字
    print(a, end=",")
    a, b = b, a + b

valid_input = False
while not valid_input:  # 要求用户输入一个整数，直到用户输入一个有效的整数为止
    try:
        num = int(input("请输入一个整数："))
        valid_input = True
    except ValueError:
        print("输入错误，请输入一个整数！")

x = 0
while x < 10:  # 打印出1到10之间的奇数
    x += 1
    if x % 2 == 0:
        continue
    print(x)



#【例1-15】函数应用示例。依次输入以下代码，并观察输出结果。

# 不含参函数、含参函数、默认参数、两种可变参函数、匿名函数

def greet_print():
    print("Hello,world!")

greet_print()  # 调用函数


# 形式参数 实际参数
def greet(name):
    print("Hello," + name + "!")

greet("Alice")  # 调用函数，并传入参数

def add(a, b, c):
    return a + b +c

result = add(3, 5, 9)  # 调用函数，并接收返回值
print(result)


def greet(name="world"):
    print("Hello," + name + "!")

greet()  # 没有传入参数时，使用默认值
greet("Alice")  # 传入参数时，使用传入的值



# arguments
# keyword arguments
def sum_all(*args):
    total = 0
    for num in args:
        total += num
    return total

print(sum_all(1, 2, 3, 4, 5))  # 输出：15
print(sum_all(1, 2, 3, 4, 5,6,7,8,9)) 

def test_kwargs(first, *args, **kwargs):
   print('Required argument: ', first)
   print(type(kwargs))
   for v in args:
      print ('Optional argument (args): ', v)
   for k, v in kwargs.items():
      print ('Optional argument %s (kwargs): %s' % (k, v))

test_kwargs(1, 2, 3, 4, k1=5, k2=6)
# key-value  k1 = 5
# 键值对


def add1(x,y):
    add_numbers = x+y
    return add_numbers

#【例1-16】匿名函数示例，演示一个接受两个参数并返回它们的和的匿名函数。
add_numbers = lambda x, y: x + y  # 定义匿名函数，接受参数x和y，并返回它们的和

# 调用匿名函数
result = add_numbers(3, 5)  # 将匿名函数赋值给变量add_numbers，并调用该函数
print("The sum is:", result)

numbers = [1, 2, 3, 4, 5]
squared_numbers = map(lambda x: x ** 2, numbers)
print(list(squared_numbers))
