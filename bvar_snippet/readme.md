
# C++模板小技巧：bvar中‌动态检测模板参数Op操作符是否实现加法

## ‌动态检测模板参数Op操作符是否实现加法

之前我们讲过bvar代码，今天分享一个关于c++模板的用法。‌动态检测模板参数Op操作符是否实现了加法行为。


```c++
// brpc/src/bvar/detail/series.h
template <typename T, typename Op, typename Enabler = void>
struct DivideOnAddition
{
    static void inplace_divide(T &obj, const Op &, int)
    {
        // 默认不执行任何操作
    }
};

template <typename T, typename Op>
struct ProbablyAddtition
{
    ProbablyAddtition(const Op &op)
    {
        T res(32);
        call_op_returning_void(op, res, T(64)); // res = op(32, 64)
        _ok = (res == T(96));                   // 只有加法才会得到 32+64=96
    }
    operator bool() const { return _ok; }

private:
    bool _ok;
};

// 整数类型的特化
template <typename T, typename Op>
struct DivideOnAddition<T, Op, typename butil::enable_if<butil::is_integral<T>::value>::type>
{
    static void inplace_divide(T &obj, const Op &op, int number)
    {
        static ProbablyAddtition<T, Op> probably_add(op);
        if (probably_add)
        {
            obj = (T)round(obj / (double)number); // 整数除法四舍五入
        }
    }
};

// 浮点类型的特化
template <typename T, typename Op>
struct DivideOnAddition<T, Op, typename butil::enable_if<butil::is_floating_point<T>::value>::type>
{
    static void inplace_divide(T &obj, const Op &op, int number)
    {
        static ProbablyAddtition<T, Op> probably_add(op);
        if (probably_add)
        {
            obj /= number; // 浮点直接除法
        }
    }
};
```

在代码中，ProbablyAddtition 的作用是 ‌动态检测 Op 操作符是否实现了加法行为‌，并根据检测结果决定是否执行除法操作。

它是如何验证是否实现加法的呢？
+ 在构造 ProbablyAddtition 时，会创建一个初始值为 32 的 T 类型对象 res。
+ 调用 call_op_returning_void(op, res, T(64))，假设这会将 res 与 64 进行 Op 操作。
+ 如果 Op 是加法，res 会变为 32 + 64 = 96，此时 _ok 被设为 true。如果 Op 不是加法（如乘法），res 会变成其他值，_ok 设为 false。

只有检测到 Op 是加法时（probably_add 为 true），才会对输入 obj 执行除法操作。

```c++
static ProbablyAddtition<T, Op> probably_add(op);
```
由于 probably_add 是静态变量，‌针对每个 T 和 Op 类型组合，检测只会进行一次‌。首次调用 inplace_divide 时检测，后续直接复用结果。

## bvar中的应用
上面的代码在bvar中是用来实现SeriesBase类的append_second方法。

对于SeriesBase类的append_second方法，它首先将value赋值给_data.second(_nsecond)，然后_nsecond自增1。如果_nsecond等于60，则将_data.second(0)的值赋给tmp，并遍历从1到59的索引，将_data.second(i)的值与tmp进行op操作，并将结果赋给tmp。最后，调用DivideOnAddition<T, Op>::inplace_divide(tmp, op, 60)对tmp进行除法操作。然后对分钟进行赋值，并调用append_minute方法。

也就是假如统计61秒的时候，就把前60秒的一个平均值计算出来，并赋值给分钟。超过60分钟，就平均统计，进位到小时，以此类推。超过24小时，就平均统计，进位到天，以此类推。

```c++
template <typename T, typename Op>
void SeriesBase<T, Op>::append_second(const T& value, const Op& op) {
    _data.second(_nsecond) = value;
    ++_nsecond;
    if (_nsecond >= 60) {
        _nsecond = 0;
        T tmp = _data.second(0);
        for (int i = 1; i < 60; ++i) {
            call_op_returning_void(op, tmp, _data.second(i));
        }
        DivideOnAddition<T, Op>::inplace_divide(tmp, op, 60);
        append_minute(tmp, op);
    }
}

template <typename T, typename Op>
class SeriesBase {
public:
    explicit SeriesBase(const Op& op)
        : _op(op)
        , _nsecond(0)
        , _nminute(0)
        , _nhour(0)
        , _nday(0) {
        pthread_mutex_init(&_mutex, NULL);
    }
    ~SeriesBase() {
        pthread_mutex_destroy(&_mutex);
    }

    void append(const T& value) {
        BAIDU_SCOPED_LOCK(_mutex);
        return append_second(value, _op);
    }

private:
    void append_second(const T& value, const Op& op);
    ...

    struct Data {
    public:
        Data() {
            // is_pod does not work for gcc 3.4
            if (butil::is_integral<T>::value ||
                butil::is_floating_point<T>::value) {
                memset(static_cast<void*>(_array), 0, sizeof(_array));
            }
        }
        
        T& second(int index) { return _array[index]; }
        const T& second(int index) const { return _array[index]; }

        T& minute(int index) { return _array[60 + index]; }
        const T& minute(int index) const { return _array[60 + index]; }

        T& hour(int index) { return _array[120 + index]; }
        const T& hour(int index) const { return _array[120 + index]; }

        T& day(int index) { return _array[144 + index]; }
        const T& day(int index) const { return _array[144 + index]; }
    private:
        T _array[60 + 60 + 24 + 30];
    };

protected:
    Op _op;
    char _nsecond;
    char _nminute;
    char _nhour;
    char _nday;
    Data _data;
};
```

### 通过一个例子看它的用法

下面是一个简单的例子，展示了如何使用这个模板：

```c++
#include <iostream>
#include <cmath>
#include <functional> // for std::plus, std::multiplies

// 模拟 is_integral 和 is_floating_point：判断一个类型是否为整数或浮点数
// bvar实现定义在：brpc/src/butil/type_traits.h
namespace butil
{
    template <typename T>
    struct is_integral
    {
        static const bool value = std::is_integral<T>::value;
    };

    template <typename T>
    struct is_floating_point
    {
        static const bool value = std::is_floating_point<T>::value;
    };

    template <bool B, typename T = void>
    struct enable_if
    {
    };

    template <typename T>
    struct enable_if<true, T>
    {
        typedef T type;
    };
} // namespace butil

// 模拟 call_op_returning_void：将 op 作用于 a 和 b，结果存入 a
template <typename Op, typename T>
void call_op_returning_void(const Op &op, T &a, const T &b)
{
    a = op(a, b);
}

// -------------------- 原始代码中的模板定义 --------------------


// -------------------- 测试用例 --------------------
int main()
{
    // 测试1：Op 是加法（std::plus<int>）
    {
        int x = 100;
        DivideOnAddition<int, std::plus<int>>::inplace_divide(x, std::plus<int>(), 3);
        std::cout << "Test1 (int, add): x = " << x << std::endl; // 应输出 33（100/3≈33.33四舍五入）
    }

    // 测试2：Op 是乘法（std::multiplies<int>）
    {
        int x = 100;
        DivideOnAddition<int, std::multiplies<int>>::inplace_divide(x, std::multiplies<int>(), 3);
        std::cout << "Test2 (int, multiply): x = " << x << std::endl; // 应保持100（不执行除法）
    }

    // 测试3：Op 是加法（std::plus<double>）
    {
        double x = 100.0;
        DivideOnAddition<double, std::plus<double>>::inplace_divide(x, std::plus<double>(), 3);
        std::cout << "Test3 (double, add): x = " << x << std::endl; // 应输出 33.3333（100/3）
    }

    // 测试4：Op 是乘法（std::multiplies<double>）
    {
        double x = 100.0;
        DivideOnAddition<double, std::multiplies<double>>::inplace_divide(x, std::multiplies<double>(), 3);
        std::cout << "Test4 (double, multiply): x = " << x << std::endl; // 应保持100.0（不执行除法）
    }

    return 0;
}

// Test1 (int, add): x = 33
// Test2 (int, multiply): x = 100
// Test3 (double, add): x = 33.3333
// Test4 (double, multiply): x = 100
```
