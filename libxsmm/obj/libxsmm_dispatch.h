#if !defined(_WIN32)
{ static const char *const build_state =
#   include "../.state"
  ;
  internal_build_state = build_state;
}
#endif
