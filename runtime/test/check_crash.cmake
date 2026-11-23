# Run tiny_crash under the loader in one of its modes (-DMODE=filter|terminate) and check from outside that the
# title's own handler ran: the run must end with the handler's exit code (-DEXPECT) and print the handler's line
# (-DLINE). A fault that ends the process instead gives 0xC0000005 or 0xE06D7363 and no such line.
# Regular expressions here avoid "\.": it is not an escape CMake accepts in a quoted argument.
get_filename_component(guest_dir "${GUEST}" DIRECTORY)
get_filename_component(guest_name "${GUEST}" NAME)
execute_process(
  COMMAND ${RUNNER} "${LOADER}" --run "${guest_name}" -- "${MODE}"
  WORKING_DIRECTORY "${guest_dir}"
  TIMEOUT 90
  RESULT_VARIABLE code OUTPUT_VARIABLE output ERROR_VARIABLE output)
string(REPLACE "\r\n" "\n" output "${output}")
string(FIND "${output}" "\n${LINE}\n" line_at)
if(NOT code STREQUAL "${EXPECT}" OR line_at EQUAL -1 OR output MATCHES "tiny_crash: FAIL")
  message(FATAL_ERROR "tiny_crash ${MODE} returned ${code}, expected ${EXPECT} and the line \"${LINE}\"\n${output}")
endif()
