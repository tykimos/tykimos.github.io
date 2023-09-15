\paper { 
  indent = 0\mm
}

\header{
  title = "나 비 야"
  composer = "LSTM v1"
}

melody = \relative c'' {
\clef treble
\key c \major
\autoBeamOff
\time 2/4

g8 e8 e4 
f8 f4 r8
d4 e8 r8
g4 d4 
\break
d4 e8 r8
g4 d4 
d4 e8 r8
g4 d4 
\break
d4 e8 r8
g4 d4 
d4 e8 r8
g4 d4 
\break
d4 e8 r8
g4 d4
d4 e8 r8
g4 d4 
\break
d4 e8 r8
g4 d4 
d4 e8 r8
g4 d4 
\break
d4 e8 r8
g4 d4 
d4 e8 r8
g4 d4 
\break
d4 e8 r8
g4 d4 
d4 r4
}

\addlyrics {

#"g8" #"e8" #"e4"
#"f8" #"f4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4" #"e8"
#"g4" #"d4"
#"d4"
}

\score {
  \new Staff \melody
  \layout { }
  \midi { }
}

\version "2.18.2"  % necessary for upgrading to future LilyPond versions.