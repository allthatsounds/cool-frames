classdef PropFilterCenterFrequency < matlab.unittest.TestCase
%PROPFILTERCENTERFREQUENCY  Centre-frequency accuracy for all filter constructors.
%
%   For every constructor, specifying fc should place the peak of |H(L)|
%   at the DFT bin closest to fc.
%
%   LTFAT frequency convention: fc in [0, 2], where fc=1 is Nyquist.
%   Peak bin (1-indexed) = round(fc / 2 * L) + 1.
%
%   Tolerance: ±3 bins (covers rounding at the bin boundaries).

    properties
        L   = 1024     % long enough for good frequency resolution
        tol_bins = 3   % allowed bin error
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
        end
    end

    % ── blfilter centre frequency ─────────────────────────────────────────────
    methods (Test)

        function testBlFilterCentreFreqSweep(tc)
            for fc = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
                g = blfilter('hann', 0.08, fc, 'peak');
                H = comp_transferfunction(g, tc.L);
                [~, peak_bin] = max(abs(H));
                expected      = round(fc / 2 * tc.L) + 1;
                tc.verifyLessThan(abs(peak_bin - expected), tc.tol_bins, ...
                    sprintf('blfilter fc=%.2f: peak_bin=%d, expected=%d', ...
                    fc, peak_bin, expected));
            end
        end

        function testBlFilterHzCentreFreq(tc)
            fs  = 8000;
            fcs = [500, 1000, 2000, 3000];
            for k = 1:numel(fcs)
                fc_hz = fcs(k);
                g     = blfilter('hann', 0.08, fc_hz, 'peak', 'fs', fs);
                H     = comp_transferfunction(g, tc.L);
                [~, peak_bin] = max(abs(H));
                fc_norm   = fc_hz / (fs/2);
                expected  = round(fc_norm / 2 * tc.L) + 1;
                tc.verifyLessThan(abs(peak_bin - expected), tc.tol_bins, ...
                    sprintf('blfilter fc_hz=%d: peak off by %d bins', ...
                    fc_hz, abs(peak_bin - expected)));
            end
        end

    end

    % ── freqfilter centre frequency ───────────────────────────────────────────
    methods (Test)

        function testFreqFilterCentreFreqSweep(tc)
            for fc = [0.1, 0.2, 0.3, 0.5, 0.7]
                g = freqfilter('gauss', 0.05, fc, 'peak');
                H = comp_transferfunction(g, tc.L);
                [~, peak_bin] = max(abs(H));
                expected      = round(fc / 2 * tc.L) + 1;
                tc.verifyLessThan(abs(peak_bin - expected), tc.tol_bins, ...
                    sprintf('freqfilter fc=%.2f: peak_bin=%d, expected=%d', ...
                    fc, peak_bin, expected));
            end
        end

        function testFreqFilterGammatoneCentreFreq(tc)
            for fc = [0.2, 0.4, 0.6]
                g = freqfilter('gammatone', 0.05, fc, 'peak');
                H = comp_transferfunction(g, tc.L);
                [~, peak_bin] = max(abs(H));
                expected      = round(fc / 2 * tc.L) + 1;
                tc.verifyLessThan(abs(peak_bin - expected), tc.tol_bins, ...
                    sprintf('freqfilter gammatone fc=%.2f: peak off', fc));
            end
        end

    end

    % ── firfilter centre frequency ────────────────────────────────────────────
    methods (Test)

        function testFirFilterCentreFreqSweep(tc)
            for fc = [0.1, 0.25, 0.4]
                g = firfilter('hann', 128, fc, 'peak');
                H = comp_transferfunction(g, tc.L);
                [~, peak_bin] = max(abs(H));
                expected      = round(fc / 2 * tc.L) + 1;
                tc.verifyLessThan(abs(peak_bin - expected), tc.tol_bins, ...
                    sprintf('firfilter fc=%.2f: peak off', fc));
            end
        end

    end

    % ── biquadfilter centre frequency ─────────────────────────────────────────
    methods (Test)

        function testBiquadFilterCentreFreqSweep(tc)
            % biquadfilter uses conjugate poles (real-coefficient IIR), so
            % the magnitude response is symmetric: |H(omega)| = |H(-omega)|.
            % Both the positive-frequency pole and its negative-frequency
            % mirror have essentially equal |H| magnitude; floating-point
            % rounding can make either one the global argmax.  Search only
            % the positive-frequency half (bins 1..L/2+1, i.e., DC to
            % Nyquist) to locate the intended resonance unambiguously.
            for fc = [0.1, 0.2, 0.3, 0.4, 0.5]
                g = biquadfilter(fc, 0.02);
                H = g.H(tc.L);
                H_pos = H(1 : tc.L/2 + 1);   % positive-frequency half
                [~, peak_bin] = max(abs(H_pos));
                expected      = round(fc / 2 * tc.L) + 1;
                tc.verifyLessThan(abs(peak_bin - expected), tc.tol_bins, ...
                    sprintf('biquadfilter fc=%.2f: peak_bin=%d, expected=%d', ...
                    fc, peak_bin, expected));
            end
        end

    end

    % ── DC filter: peak at bin 1 ──────────────────────────────────────────────
    methods (Test)

        function testBlFilterDCPeak(tc)
            g = blfilter('hann', 0.1, 0, 'peak');
            H = comp_transferfunction(g, tc.L);
            [~, peak_bin] = max(abs(H));
            tc.verifyEqual(peak_bin, 1, ...
                'DC filter: peak should be at bin 1');
        end

        function testFreqFilterDCPeak(tc)
            g = freqfilter('gauss', 0.05, 0, 'peak');
            H = comp_transferfunction(g, tc.L);
            [~, peak_bin] = max(abs(H));
            tc.verifyEqual(peak_bin, 1, ...
                'DC freqfilter: peak should be at bin 1');
        end

    end

    % ── Monotone shift with increasing fc ────────────────────────────────────
    methods (Test)

        function testBlFilterMonotonePeakWithIncreasingFc(tc)
            % Peak bin should increase monotonically as fc increases
            fcs      = 0.1 : 0.05 : 0.7;
            peak_bins = zeros(size(fcs));
            for k = 1:numel(fcs)
                g = blfilter('hann', 0.06, fcs(k), 'peak');
                H = comp_transferfunction(g, tc.L);
                [~, peak_bins(k)] = max(abs(H));
            end
            diffs = diff(peak_bins);
            tc.verifyTrue(all(diffs >= 0), ...
                'Peak bin should be non-decreasing as fc increases');
        end

    end

end
