load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")
load("//tools/build_defs/license:license.bzl", "license")

package(
    default_applicable_licenses = [":license"],
    default_visibility = ["//visibility:private"],
)

license(
    name = "license",
    package_name = "natural_plan",
)

licenses(["notice"])

exports_files(["LICENSE"])

pytype_strict_binary(
    name = "evaluate_trip_planning",
    srcs = ["evaluate_trip_planning.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)

pytype_strict_binary(
    name = "evaluate_meeting_planning",
    srcs = ["evaluate_meeting_planning.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)

pytype_strict_binary(
    name = "evaluate_calendar_scheduling",
    srcs = ["evaluate_calendar_scheduling.py"],
    deps = [
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
    ],
)
