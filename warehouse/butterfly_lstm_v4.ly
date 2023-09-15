\paper { 
  indent = 0\mm
}

\header{
  title = "나 비 야"
  composer = "LSTM v3"
}

melody = \relative c'' {
\clef treble
\key c \major
\autoBeamOff
\time 2/4

g8 e8 e4 
f8 d8 d4 
c8 d8 e8 f8 g8 g8 g4 g8 e8 e8 e8 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4 d8 d8 d8 d8 d8 e8 f4 e8 e8 e8 e8 e8 f8 g4 g8 e8 e4 f8 d8 d4 c8 e8 g8 g8 e8 e8 e4
}

\score {
  \new Staff \melody
  \layout { }
  \midi { }
}

\version "2.18.2"  % necessary for upgrading to future LilyPond versions.