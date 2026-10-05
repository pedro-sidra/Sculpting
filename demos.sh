IDs=(
"tv5y9vop"
"ea7k5e6a"
#   "gsntq83r"
#   "pqm95tdv"
#   "qs4z9sr3"
#   "kchltoey"
#   "pw86ei5f"
#   "55y0nl4r"
#   "kbjmpqnq"
#   "bqgxvp4u"
#   "ke3or8ft"
#   "kjat7htt"
#   "hbz5jpv6"
#   "2849ch9g"
#   "db079bj7"
#   "sjm7zxrx"
#   "8ibyxmtf"
)
for i in "${IDs[@]}"; do
    python tools/generate_inference_sample_seg.py $i
done