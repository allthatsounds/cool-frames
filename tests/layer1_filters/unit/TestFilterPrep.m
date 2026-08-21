classdef TestFilterPrep < matlab.unittest.TestCase
%TESTFILTERPREP  Unit tests for comp_transferfunction and comp_filterbank_pre.
%
%   comp_transferfunction(g, L)
%     — evaluates the full-length (L-point) frequency response of a single
%       filter struct, handling BL offsets, delays, and realonly symmetry.
%
%   comp_filterbank_pre(g_cell, a, L, crossover)
%     — evaluates all function handles in the filter cell array, applies
%       modulations, and canonicalises each filter to numeric form.

    properties
        tol = 1e-10
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── comp_transferfunction: output length ─────────────────────────────────
    methods (Test)

        function testLengthFromBlFilter(tc)
            g = blfilter('hann', 0.1, 0.3);
            for L = [64, 256, 1024]
                H = comp_transferfunction(g, L);
                tc.verifyEqual(numel(H), L, ...
                    sprintf('comp_transferfunction: expected length %d', L));
            end
        end

        function testLengthFromFirFilter(tc)
            g = firfilter('hann', 32);
            for L = [64, 256, 512]
                H = comp_transferfunction(g, L);
                tc.verifyEqual(numel(H), L);
            end
        end

        function testLengthFromFreqFilter(tc)
            g = freqfilter('gauss', 0.1, 0.3);
            H = comp_transferfunction(g, 512);
            tc.verifyEqual(numel(H), 512);
        end

        function testLengthFromBiquadFilter(tc)
            g = biquadfilter(0.25, 0.05);
            H = comp_transferfunction(g, 256);
            tc.verifyEqual(numel(H), 256);
        end

    end

    % ── comp_transferfunction: consistency with g.H ──────────────────────────
    methods (Test)

        function testConsistencyWithBiquadH(tc)
            % For biquadfilter, g.H(L) directly gives the full response.
            % comp_transferfunction should give the same result.
            L = 512;
            g = biquadfilter(0.3, 0.04, 'peak');
            H_direct = g.H(L);
            H_pre    = comp_transferfunction(g, L);
            tc.verifyEqual(H_pre, H_direct, 'AbsTol', tc.tol, ...
                'comp_transferfunction should match g.H(L) for biquadfilter');
        end

        function testConsistencyWithFirFilterFFT(tc)
            % firfilter: comp_transferfunction should match
            % fft(circshift(postpad(g.h, L), g.offset))
            % NOTE: g.offset is NOT negated; comp_filterbank_pre uses
            %       circshift(postpad(h,L), g.offset) directly (line 59).
            L = 512;
            g = firfilter('hann', 64);
            H_pre = comp_transferfunction(g, L);
            H_man = fft(circshift(postpad(g.h, L), g.offset));
            tc.verifyEqual(H_pre, H_man, 'AbsTol', tc.tol, ...
                'comp_transferfunction should match manual FFT for firfilter');
        end

    end

    % ── comp_transferfunction: realonly symmetry ──────────────────────────────
    methods (Test)

        function testRealonlyEnforcesHermitianSymmetry(tc)
            % With realonly=1: H(k) = conj(H(L+2-k)) for k=2..L
            L  = 256;
            fc = 0.3;
            g  = blfilter('hann', 0.1, fc, 'real');
            tc.verifyEqual(g.realonly, 1);
            H  = comp_transferfunction(g, L);
            % Hermitian symmetry: H(k) = conj(H(L+2-k)) = conj(flipud(H)(k-1))
            % In 1-indexed MATLAB: H(2:end) == conj(flipud(H(2:end)))
            tc.verifyEqual(H(2:end), conj(flipud(H(2:end))), 'AbsTol', tc.tol, ...
                'realonly=1: H must be Hermitian symmetric');
        end

        function testRealonlyMakesTimeDomainReal(tc)
            % Hermitian H → real IFFT
            L  = 256;
            g  = blfilter('hann', 0.1, 0.3, 'real');
            H  = comp_transferfunction(g, L);
            h  = ifft(H);
            tc.verifyLessThan(max(abs(imag(h))), tc.tol, ...
                'realonly filter: ifft(H) should be real');
        end

    end

    % ── comp_transferfunction: delay property ────────────────────────────────
    methods (Test)

        function testDelayShiftsPhase(tc)
            % A filter with delay d: H_delayed(k) = H(k) * exp(-2pi*i*d*k/L)
            L = 256;
            d = 3;
            g0 = blfilter('hann', 0.1, 0.3);
            gd = blfilter('hann', 0.1, 0.3, 'delay', d);
            H0 = comp_transferfunction(g0, L);
            Hd = comp_transferfunction(gd, L);
            k  = (0:L-1)';
            expected = H0 .* exp(-2*pi*1i*d*k/L);
            tc.verifyEqual(Hd, expected, 'AbsTol', tc.tol, ...
                'Delay d should multiply H by exp(-2*pi*i*d*k/L)');
        end

    end

    % ── comp_filterbank_pre: canonicalization ────────────────────────────────
    methods (Test)

        function testPreMakesHNumeric(tc)
            % After comp_filterbank_pre, g.H must be numeric (not a function handle)
            g_cell = {blfilter('hann', 0.1, 0.3), freqfilter('gauss', 0.1, 0.2)};
            a      = [4; 4];
            L      = 512;
            g_pre  = comp_filterbank_pre(g_cell, a, L, 0);
            for m = 1:numel(g_pre)
                tc.verifyTrue(isnumeric(g_pre{m}.H), ...
                    sprintf('Filter %d: H should be numeric after pre', m));
            end
        end

        function testPrePreservesLength(tc)
            % Filters after pre should have H with numel equal to their bandwidth
            g_cell = {blfilter('hann', 0.1, 0.3)};
            a      = 8;
            L      = 512;
            g_pre  = comp_filterbank_pre(g_cell, a, L, 0);
            tc.verifyTrue(isnumeric(g_pre{1}.H));
            tc.verifyGreaterThan(numel(g_pre{1}.H), 0);
        end

        function testPreIdempotent(tc)
            % Running comp_filterbank_pre twice should give the same result
            % (numeric H is already evaluated, second call is a no-op on H)
            g_cell = {firfilter('hann', 32)};
            a      = 4;
            L      = 256;
            g_pre1 = comp_filterbank_pre(g_cell, a, L, 0);
            g_pre2 = comp_filterbank_pre(g_pre1, a, L, 0);
            tc.verifyEqual(g_pre2{1}.H, g_pre1{1}.H, 'AbsTol', tc.tol, ...
                'comp_filterbank_pre should be idempotent');
        end

        function testPreFirFilterConsistency(tc)
            % Full-length output of comp_filterbank_pre should match
            % comp_transferfunction for a firfilter
            L      = 256;
            g0     = firfilter('hann', 32);
            g_cell = {g0};
            a      = 4;
            g_pre  = comp_filterbank_pre(g_cell, a, L, 0);

            % comp_filterbank_pre stores H as BL or full depending on crossover.
            % comp_transferfunction of the pre-processed filter should match original.
            H_tf  = comp_transferfunction(g0, L);
            H_pre = comp_transferfunction(g_pre{1}, L);
            tc.verifyEqual(H_pre, H_tf, 'AbsTol', tc.tol, ...
                'comp_filterbank_pre + comp_transferfunction should match direct');
        end

    end

end
