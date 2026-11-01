#
# Copyright (c) 2026, Daniel Garcia Peña (iamdgarcia)
#
# SPDX-License-Identifier: MIT
#

"""Tests for prompt building and prefix alignment."""

from pipecat_speculative_prefill.prompt import build_messages, longest_common_prefix_len


class TestLongestCommonPrefixLen:
    def test_identical(self):
        assert longest_common_prefix_len([1, 2, 3], [1, 2, 3]) == 3

    def test_prefix(self):
        assert longest_common_prefix_len([1, 2], [1, 2, 3, 4]) == 2

    def test_divergence_midway(self):
        assert longest_common_prefix_len([1, 2, 9], [1, 2, 3]) == 2

    def test_no_overlap(self):
        assert longest_common_prefix_len([1], [2]) == 0

    def test_empty(self):
        assert longest_common_prefix_len([], [1, 2]) == 0
        assert longest_common_prefix_len([], []) == 0


class TestBuildMessages:
    def test_system_instruction_prepended(self):
        msgs = build_messages([{"role": "user", "content": "hi"}], "be brief")
        assert msgs == [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]

    def test_developer_role_mapped_to_system(self):
        msgs = build_messages([{"role": "developer", "content": "intro"}], None)
        assert msgs == [{"role": "system", "content": "intro"}]

    def test_partial_appended_as_user_turn(self):
        msgs = build_messages([{"role": "system", "content": "s"}], None, partial_user_text="hel")
        assert msgs[-1] == {"role": "user", "content": "hel"}

    def test_multimodal_content_flattened_to_text(self):
        content = [
            {"type": "text", "text": "what is this"},
            {"type": "image_url", "image_url": {"url": "http://x/y.png"}},
        ]
        msgs = build_messages([{"role": "user", "content": content}], None)
        assert msgs == [{"role": "user", "content": "what is this"}]
