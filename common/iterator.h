#pragma once

#include <tuple>
#include <utility>

#include <common/log.h>

namespace chainer_compiler {

// `C1` and `C2` are containers.
template <class C1, class C2>
std::vector<std::tuple<typename C1::value_type, typename C2::value_type>> Zip(const C1& c1, const C2& c2) {
    typedef typename C1::value_type E1;
    typedef typename C2::value_type E2;
    CHECK_EQ(c1.size(), c2.size());
    std::vector<std::tuple<E1, E2>> results;
    typename C1::const_iterator iter1 = c1.begin();
    for (const E2& v2 : c2) {
        results.emplace_back(*iter1, v2);
        ++iter1;
    }
    return results;
}

template <class C>
class Enumerator {
public:
    struct Entry {
        Entry(const typename C::value_type& v, size_t i)
            : value(v), index(i) {
        }

        const typename C::value_type& value;
        size_t index;
    };

    class Iterator {
    public:
        Iterator(const C& c, bool end)
            : c_(c) {
            if (end) {
                iter_ = c.end();
                index_ = c.size();
            } else {
                iter_ = c.begin();
                index_ = 0;
            }
        }

        Entry operator*() const {
            return Entry(*iter_, index_);
        }

        Iterator& operator++() {
            CHECK(iter_ != c_.end());
            ++iter_;
            ++index_;
            return *this;
        }

        bool operator!=(const Iterator& r) const {
            return iter_ != r.iter_;
        }

    private:
        const C& c_;
        typename C::const_iterator iter_;
        size_t index_;
    };

    Enumerator(const C& c)
        : c_(c) {
    }

    Iterator begin() const {
        return Iterator(c_, false);
    }

    Iterator end() const {
        return Iterator(c_, true);
    }

private:
    const C& c_;
};

// `C` is a container.
template <class C>
Enumerator<C> Enumerate(const C& c) {
    return Enumerator<C>(c);
}

}  // namespace
