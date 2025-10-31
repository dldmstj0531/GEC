# How to use?
## download_shard.ipynb 
캐글 데이터셋에서 n-of-00010번 샤드 다운로드
```
name                            size  creationDate                
------------------------  ----------  --------------------------  
edits.tsv-00000-of-00010  2992601947  2021-05-26 18:59:22.840000  
edits.tsv-00001-of-00010  2990782945  2021-05-26 19:00:11.393000  
edits.tsv-00002-of-00010  2991847440  2021-05-26 19:00:57.927000  
edits.tsv-00003-of-00010  2990847051  2021-05-26 19:01:38.295000  
edits.tsv-00004-of-00010  2991463256  2021-05-26 19:02:19.945000  
edits.tsv-00005-of-00010  2992038211  2021-05-26 19:03:00.175000  
edits.tsv-00006-of-00010  2992255660  2021-05-26 19:03:41.023000  
edits.tsv-00007-of-00010  2993397600  2021-05-26 19:04:18.859000  
edits.tsv-00008-of-00010  2991033963  2021-05-26 19:05:03.250000  
edits.tsv-00009-of-00010  2991637077  2021-05-26 19:05:42.223000  
```
허깅페이스 allenai/c4에서 json.gz 파일 다운로드
```
-rw-r--r-- 1 root root 305M Oct 25 23:51 c4-train.00000-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00001-of-01024.json.gz
-rw-r--r-- 1 root root 305M Oct 26 00:09 c4-train.00002-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00003-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00004-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00005-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00006-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00007-of-01024.json.gz
-rw-r--r-- 1 root root 304M Oct 26 00:09 c4-train.00008-of-01024.json.gz
```
## c4_200m_pairs_pipeline_notebook.ipynb
download_shard.ipynb 에서 받은 SHARD_PATH, JSON_DIR을 입력받아 sentence_pairs.tsv 생성
```
[INFO] using edits: /content/tmp_edits.tsv | size: 2990847051
Loading C4_200M target sentence hashes from '/content/tmp_edits.tsv'...
Searching for 19928615 target sentences in the C4 dataset...
-- 0 C4 examples done, 19928615 sentences still to be found
-- 100000 C4 examples done, 19923699 sentences still to be found
-- 200000 C4 examples done, 19918547 sentences still to be found
-- 300000 C4 examples done, 19913411 sentences still to be found
...
-- 82400000 C4 examples done, 15778440 sentences still to be found
-- 82500000 C4 examples done, 15773415 sentences still to be found
-- 82600000 C4 examples done, 15768463 sentences still to be found
Found 4167881 target sentences (15765111 not found).
Writing C4_200M sentence pairs to '/content/out/target_sentences.tsv-00004-of-00010'...
lines: 4167881 /content/out/sentence_pairs.tsv-00004-of-00010
There is I believe it's all you need to know time for being beeing.	I believe that's all you need to know for the time-being.
how certain are you that new version of DMN will not invalid your models?	How certain are you that new versions of DMN will not invalidate your models?
trainig, we return call your call back.	training, we will return your call.
```

## preprocess_data_01.ipynb
[noise,clean] sentence_pairs를 가진 csv파일 입력받아 GEC_tag 붙이기
```
The size of raw dataset is 20000
20000it [00:13, 1536.77it/s]
Overall extracted 20000. Original TP 19882. Original TN 118
샘플 전처리 완료. 출력 파일 목록:
-rw-r--r-- 1 root root 11M Oct 27 06:03 /content/processed.dry
processed.dry :
$STARTSEPL|||SEPR$KEEP MuchSEPL|||SEPR$DELETE manySEPL|||SEPR$TRANSFORM_CASE_CAPITAL brandsSEPL|||SEPR$KEEP andSEPL|||SEPR$KEEP sellersSEPL|||SEPR$KEEP stillSEPL|||SEPR$KEEP inSEPL|||SEPR$KEEP theSEPL|||SEPR$KEEP market.SEPL|||SEPR$KEEP
$STARTSEPL|||SEPR$KEEP FairySEPL|||SEPR$KEEP OrSEPL|||SEPR$KEEP Not,SEPL|||SEPR$KEEP I'mSEPL|||SEPR$KEEP theSEPL|||SEPR$KEEP Godmother:SEPL|||SEPR$KEEP noSEPL|||SEPR$REPLACE_Not justSEPL|||SEPR$APPEND_a look,SEPL|||SEPR$KEEP butSEPL|||SEPR$KEEP mySEPL|||SEPR$KEEP outfitSEPL|||SEPR$KEEP forSEPL|||SEPR$KEEP takingSEPL|||SEPR$APPEND_on theSEPL|||SEPR$KEEP partSEPL|||SEPR$REPLACE_role asSEPL|||SEPR$KEEP godmother.SEPL|||SEPR$KEEP
[Line 1]
  Much                  <--  KEEP
  many                  <--  DELETE
  brands                <--  TRANSFORM_CASE_CAPITAL
  and                   <--  KEEP
  sellers               <--  KEEP
  still                 <--  KEEP
  in                    <--  KEEP
  the                   <--  KEEP
  market.               <--  KEEP
[Line 2]
  Fairy                 <--  KEEP
  Or                    <--  KEEP
  Not,                  <--  KEEP
  I'm                   <--  KEEP
  the                   <--  KEEP
  Godmother:            <--  KEEP
  no                    <--  KEEP
  just                  <--  REPLACE_Not
  look,                 <--  APPEND_a
  but                   <--  KEEP
  my                    <--  KEEP
  outfit                <--  KEEP
  for                   <--  KEEP
  taking                <--  KEEP
  the                   <--  APPEND_on
  part                  <--  KEEP
  as                    <--  REPLACE_role
  godmother.            <--  KEEP

```