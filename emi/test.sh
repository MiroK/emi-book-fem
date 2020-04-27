python check_sanity.py fem/prime_single.py

FILE=/home/fenics/emi-book-fem/emi/results/sanity_fem.prime_single_standard_kappa1_eps1.txt
if test -f "$FILE"; then
    echo "You are all set"
fi
