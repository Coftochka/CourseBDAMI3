cur_dir=$(dirname "$(readlink -f "$0")")
cd "$cur_dir"

find . -name "*frac*" | xargs rm -r
