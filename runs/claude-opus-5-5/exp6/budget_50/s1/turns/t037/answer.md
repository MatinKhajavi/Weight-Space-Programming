```python
src=open('/work/builder4.py').read()
# L2 opp parity: factor and bias
src=src.replace("""        for src,dst in ((EP,XE),(OP,XO)):
            W[2][src+p,dst+p]+=0.5
            for (rr,cc) in nb: W[2][src+P(rr,cc),dst+p]+=0.5
            W[2][C1+p,dst+p]+=-K""","""        for src,dst in ((EP,XE),(OP,XO)):
            W[2][src+p,dst+p]+=prm['ofac']
            for (rr,cc) in nb: W[2][src+P(rr,cc),dst+p]+=prm['ofac']
            W[2][C1+p,dst+p]+=-K; B[2][dst+p]=prm['eps']""")
src=src.replace("""            for blk,cblk in ((SW,OWN),(S3,OWN),(XE,OPP),(XO,OPP)):
                W[k][blk+p,blk+p]+=0.5
                for (rr,cc) in nb: W[k][blk+P(rr,cc),blk+p]+=0.5
                W[k][cblk+p,blk+p]+=K2; B[k][blk+p]=-K2""","""            for blk,cblk in ((SW,OWN),(S3,OWN),(XE,OPP),(XO,OPP)):
                f = prm['ofac'] if blk in (XE,XO) else 0.5
                e = prm['eps'] if blk in (XE,XO) else 0.
                W[k][blk+p,blk+p]+=f
                for (rr,cc) in nb: W[k][blk+P(rr,cc),blk+p]+=f
                W[k][cblk+p,blk+p]+=K2; B[k][blk+p]=-K2+e""")
src=src.replace("W[6][SW+q,L6_AW+p]+=-100; W[6][tch+q,L6_AP+p]+=-100",
                "W[6][SW+q,L6_AW+p]+=-100; W[6][tch+q,L6_AP+p]+=-prm['apcoef']")
assert "prm['apcoef']" in src and "prm['eps']" in src
open('/work/builder5.py','w').write(src)
exec(src)
import json
prm=json.load(open('/work/prm_v4.json'))
prm.update(ofac=1.0,eps=0.003,apcoef=1.0)
json.dump(prm,open('/work/prm_v5.json','w'))
W,B,Wo,bo=build(prm)
save_all(save,W,B,Wo,bo)
print('saved v5')
```