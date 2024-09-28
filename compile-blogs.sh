for file in blogs/*.md;
do
  filename=${file##*/}
  basefile=${filename%.md}
  pandoc "$file" -o "public/posts/$basefile.html" -s
done