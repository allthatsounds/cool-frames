classdef TestFirFilter < matlab.unittest.TestCase
%TESTFIRFILTER  Unit tests for firfilter (FIR filter constructor).
%
%   firfilter(name, M)       — time-domain FIR filter, length M, centred at DC
%   firfilter(name, M, fc)   — FIR filter at centre freq fc (normalized)

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

    % ── Struct format ────────────────────────────────────────────────────────
    methods (Test)

        function testStructHasRequiredFields(tc)
            g = firfilter('hann', 32);
            tc.verifyTrue(isfield(g, 'h'),        'Missing: h');
            tc.verifyTrue(isfield(g, 'offset'),   'Missing: offset');
            tc.verifyTrue(isfield(g, 'delay'),    'Missing: delay');
            tc.verifyTrue(isfield(g, 'realonly'), 'Missing: realonly');
        end

        function testHIsNumericVector(tc)
            g = firfilter('hann', 32);
            tc.verifyTrue(isnumeric(g.h) && isvector(g.h), ...
                'g.h must be a numeric vector');
        end

        function testScalarInputReturnsStruct(tc)
            g = firfilter('hann', 32);
            tc.verifyTrue(isstruct(g));
        end

        function testDefaultRealonlyIsZero(tc)
            g = firfilter('hann', 32);
            tc.verifyEqual(g.realonly, 0);
        end

        function testRealFlagSetsRealonly(tc)
            g = firfilter('hann', 32, 0, 'real');
            tc.verifyEqual(g.realonly, 1);
        end

    end

    % ── Impulse response length ───────────────────────────────────────────────
    methods (Test)

        function testImpulseResponseLength(tc)
            for M = [16, 32, 64, 128]
                g = firfilter('hann', M);
                tc.verifyEqual(numel(g.h), M, ...
                    sprintf('firfilter: impulse response should have length %d', M));
            end
        end

        function testSineFilterLength(tc)
            M = 48;
            g = firfilter('sine', M);
            tc.verifyEqual(numel(g.h), M);
        end

    end

    % ── Real-valued impulse response ──────────────────────────────────────────
    methods (Test)

        function testImpulseResponseIsReal(tc)
            % firwin windows are real, so h should be real (fc=0)
            g = firfilter('hann', 32);
            tc.verifyTrue(isreal(g.h), ...
                'DC-centred firfilter: impulse response should be real');
        end

    end

    % ── Energy normalization ──────────────────────────────────────────────────
    methods (Test)

        function testEnergyNormDefault(tc)
            % Default 'energy': sum(h.^2) = 1  (Parseval: ||h||^2 = ||H||^2/L)
            g = firfilter('hann', 64);
            tc.verifyEqual(sum(g.h.^2), 1, 'AbsTol', tc.tol, ...
                'firfilter energy norm: sum(h^2) should be 1');
        end

        function testEnergyNormMultipleLengths(tc)
            for M = [16, 32, 64]
                g = firfilter('hann', M);
                tc.verifyEqual(sum(g.h.^2), 1, 'AbsTol', tc.tol, ...
                    sprintf('energy norm failed for M=%d', M));
            end
        end

        function testPeakNorm(tc)
            % 'peak' for firfilter normalises the time-domain impulse response:
            % max|h| = 1.  (This differs from blfilter/freqfilter where 'peak'
            % refers to the frequency-domain magnitude.)
            g = firfilter('hann', 64, 0, 'peak');
            tc.verifyEqual(max(abs(g.h)), 1, 'AbsTol', tc.tol, ...
                'firfilter peak norm: max|h| should be 1');
        end

        function testScalParameter(tc)
            s  = 2.0;
            g1 = firfilter('hann', 32);
            g2 = firfilter('hann', 32, 0, 'scal', s);
            tc.verifyEqual(g2.h, s * g1.h, 'AbsTol', tc.tol * norm(g1.h), ...
                '''scal'' should multiply impulse response');
        end

    end

    % ── Offset / delay ────────────────────────────────────────────────────────
    methods (Test)

        function testNonCausalOffsetIsNegativeHalfLength(tc)
            % Default (non-causal): offset = delay - floor(M/2) = 0 - floor(M/2)
            M = 32;
            g = firfilter('hann', M);
            tc.verifyEqual(g.offset, -floor(M/2));
        end

        function testCausalOffsetIsZero(tc)
            g = firfilter('hann', 32, 0, 'causal');
            tc.verifyEqual(g.offset, 0);
        end

        function testDefaultDelayIsZero(tc)
            g = firfilter('hann', 32);
            tc.verifyEqual(g.delay, 0);
        end

    end

    % ── Centre frequency via comp_transferfunction ────────────────────────────
    methods (Test)

        function testPeakNearCentreFreq(tc)
            L  = 1024;
            fc = 0.25;
            g  = firfilter('hann', 64, fc, 'peak');
            H  = comp_transferfunction(g, L);
            [~, peak_bin] = max(abs(H));
            expected_bin  = round(fc / 2 * L) + 1;
            tc.verifyLessThan(abs(peak_bin - expected_bin), 5, ...
                'firfilter: peak should be near specified fc');
        end

    end

    % ── Filterbank integration ────────────────────────────────────────────────
    methods (Test)

        function testZeroInputYieldsZeroOutput(tc)
            Ls = 512;
            x  = zeros(Ls, 1);
            g  = {firfilter('hann', 32)};
            a  = 4;
            c  = filterbank(x, g, a);
            tc.verifyLessThan(max(abs(c{1})), 1e-12, ...
                'Zero input should give zero subband output');
        end

        function testOutputSubbandLength(tc)
            Ls = 512;
            rng(42);
            x  = randn(Ls, 1);
            g  = {firfilter('hann', 32)};
            a  = 4;
            c  = filterbank(x, g, a);
            tc.verifyEqual(size(c{1}, 1), ceil(Ls / a));
        end

    end

end
