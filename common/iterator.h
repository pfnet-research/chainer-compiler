#pragma once

#include <tuple>
#include <utility>

#include <common/log.h>

namespace chainer_compiler {

template <class C1, class C2>
class Zipper {
public:
    class Iterator {
    public:
        Iterator(const C1& c1, const C2& c2, bool end)
            : c1_(c1), c2_(c2) {
            CHECK_EQ(c1.size(), c2.size());
            if (end) {
                iter1_ = c1.end();
                iter2_ = c2.end();
            } else {
                iter1_ = c1.begin();
                iter2_ = c2.begin();
            }
        }

        std::tuple<typename C1::value_type, typename C2::value_type>
        operator*() const {
            return std::make_tuple(*iter1_, *iter2_);
        }

        Iterator& operator++() {
            CHECK(iter1_ != c1_.end());
            CHECK(iter2_ != c2_.end());
            ++iter1_;
            ++iter2_;
            return *this;
        }

        bool operator!=(const Iterator& r) const {
            return iter1_ != r.iter1_;
        }

    private:
        const C1& c1_;
        const C2& c2_;
        typename C1::const_iterator iter1_;
        typename C2::const_iterator iter2_;
    };

    Zipper(const C1& c1, const C2& c2)
        : c1_(c1), c2_(c2) {
    }

    Iterator begin() const {
        return Iterator(c1_, c2_, false);
    }

    Iterator end() const {
        return Iterator(c1_, c2_, true);
    }

private:
    const C1& c1_;
    const C2& c2_;
};

// `C1` and `C2` are containers.
template <class C1, class C2>
Zipper<C1, C2> Zip(const C1& c1, const C2& c2) {
    return Zipper<C1, C2>(c1, c2);
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
