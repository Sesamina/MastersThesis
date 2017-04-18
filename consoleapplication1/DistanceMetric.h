#include <algorithm>

template<class T>
struct DistanceMetric
{
	typedef T ElementType;
	typedef typename flann::Accumulator<T>::Type ResultType;
	template <typename Iterator1, typename Iterator2>
	ResultType operator()(Iterator1 a, Iterator2 b, size_t size,
		ResultType /*worst_dist*/ = -1) const
	{
		ResultType result = ResultType();
		ResultType minSum = ResultType();
		ResultType maxSum = ResultType();
		for (size_t i = 0; i < size; ++i) {
			minSum += std::min(*a, *b);
			maxSum += std::max(*a++, *b++);
		}
		result = 1 - ((1 + minSum) / (1 + maxSum));
		return result;
	}
};