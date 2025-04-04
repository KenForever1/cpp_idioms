#include <iostream>
#include <cmath>
#include <functional> // for std::plus, std::multiplies

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