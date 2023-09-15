\paper { 
  indent = 0\mm
}

\header{
  title = "나 비 야"
  composer = "MLP 3 dense"
}

melody = \relative c'' {
\clef treble
\key c \major
\autoBeamOff
\time 2/4

g8 e8 e4
f8 e8 e4 
e8 e4 g8 
g8 e4 e8
\break
e4 g8 g8
e4 e8 r8
e4 g8 g8 
e4 e8 r8
\break
e4 g8 g8
e4 e8 r8
e4 g8 g8
e4 e8 r8
e4 g8 g8
\break
e4 e8 r8
e4 g8 r8
g8 e4 e8
e4 g8 g8
\break
e4 e8 r8
e4 g8 g8
e4 e8 r8
e4 g8 r8
}

\addlyrics {

#"g8" #"e8" #"e4"
#"f8" #"e8" #"e4" 
#"e8" #"e4" #"g8"
#"g8" #"e4" #"e8"
#"e4" #"g8" #"g8"
#"e4" #"e8" #"e4"
#"g8" #"g8" #"e4"
#"e8" #"e4" #"g8"
#"g8" #"e4" #"e8"
#"e4" #"g8" #"g8"
#"e4" #"e8" #"e4"
#"g8" #"g8" #"e4"
#"e8" #"e4" #"g8"
#"g8" #"e4" #"e8"
#"e4" #"g8" #"g8"
#"e4" #"e8" #"e4"
#"g8" #"g8" #"e4"
#"e8" #"e4" #"g8"
}

\score {
  \new Staff \melody
  \layout { }
  \midi { }
}

\version "2.18.2"  % necessary for upgrading to future LilyPond versions.