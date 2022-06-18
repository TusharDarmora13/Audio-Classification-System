import os

genres = 'blues classical country disco pop hiphop metal reggae rock'
genres = genres.split()

from pydub import AudioSegment
i = 0
for g in genres:
  j=0
  print(f"{g}")
  for filename in os.listdir(os.path.join(r'C:\Users\dogra\Desktop\archive\Data\genres_original',f"{g}")):

    song  =  os.path.join(r'C:\Users\dogra\Desktop\archive\Data\genres_original/'+f"{g}",f'{filename}')
    j = j+1
    for w in range(0,10):
      i = i+1
      #print(i)
      t1 = 3*(w)*1000
      t2 = 3*(w+1)*1000
      newAudio = AudioSegment.from_wav(song)
      new = newAudio[t1:t2]
      new.export(r'C:\Users\dogra\Desktop\archive\NEW_DATA/'+f"{g}"+'/'+f"{g+str(j)+str(w)}.wav", format="wav")