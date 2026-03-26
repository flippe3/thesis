# This is a prototype on the idea of artificial curiosity, which
# is probably more a philosophy than a useful AI approach.

# Schmidhuber has written and discussed the theory of beauty and
# low-complexity art extensively. Yet this has never really been
# explored in a serious direction.

# The principles are trivial. (https://people.idsia.ch/~juergen/creativity.html)
# Let O(t) denote the state of some subjective observer O at time t.
# Let H(t) denote its history of previous actions & sensations & rewards until time t.
# O has some adaptive method for compressing H(t) or parts of it.
# Beauty B(D, O(t)) of any data D as the negative number of bits required to encode D.
# Interestingness is defined as I(D, O(t)) = B(D, O(t)) - B(D, O(t-1)).
