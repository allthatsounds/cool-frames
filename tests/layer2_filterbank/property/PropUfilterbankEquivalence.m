classdef PropUfilterbankEquivalence < matlab.unittest.TestCase
%PROPUFILTERBANKEQUIVALENCE  ufilterbank and filterbank agree for uniform hop.
%
%   For a uniform (scalar) subsampling factor a, ufilterbank and filterbank
%   must produce identical coefficients:
%
%     ufilterbank(x, g, a)[:, m]  ==  filterbank(x, g, a_vec){m}
%
%   where a_vec = a * ones(M, 1).
%
%   ufilterbank returns a 2-D matrix (L/a × M), while filterbank returns a
%   cell array of column vectors.  This test bridges the two output formats.

    properties
        p   % scalar parameters (fs, Ls)
    end

    methods (TestClassSetup)
        function setupClass(tc)
            sharedDir = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared');
            addpath(sharedDir);
            setup_filterbank_paths();
            rng(42);
            tc.p = struct('fs', 8000, 'Ls', 1024);
        end
    end

    % Helper: build a small blfilter bank with given parameters.
    methods (Access = private)
        function [g, a_vec, L] = makeBlFilterbank(tc, M, hop, bw)
            fcs = linspace(0.1, 0.9, M);
            g   = cell(1, M);
            for m = 1:M
                g{m} = blfilter('hann', bw, fcs(m), 'peak');
            end
            a_vec = hop * ones(M, 1);
            L     = filterbanklength(tc.p.Ls, a_vec);
        end
    end

    methods (Test)

        function testEquivalenceRealSignal(tc)
            [g, a_vec, L] = tc.makeBlFilterbank(8, 4, 0.15);
            a_scalar = a_vec(1);
            x = randn(L, 1);

            c_cell   = filterbank(x, g, a_vec);
            c_matrix = ufilterbank(x, g, a_scalar);

            M = numel(g);
            for m = 1:M
                err = norm(c_cell{m} - c_matrix(:, m)) / (norm(c_cell{m}) + eps);
                tc.verifyLessThan(err, 1e-12, ...
                    sprintf('Band %d (real): ufilterbank/filterbank mismatch %.2e', m, err));
            end
        end

        function testEquivalenceComplexSignal(tc)
            [g, a_vec, L] = tc.makeBlFilterbank(6, 8, 0.12);
            a_scalar = a_vec(1);
            x = randn(L,1) + 1i*randn(L,1);

            c_cell   = filterbank(x, g, a_vec);
            c_matrix = ufilterbank(x, g, a_scalar);

            M = numel(g);
            for m = 1:M
                err = norm(c_cell{m} - c_matrix(:, m)) / (norm(c_cell{m}) + eps);
                tc.verifyLessThan(err, 1e-12, ...
                    sprintf('Band %d (complex): ufilterbank/filterbank mismatch %.2e', m, err));
            end
        end

        function testUfilterbankOutputDimensions(tc)
            % ufilterbank(x, g, a) must return an (L/a) × M matrix.
            M  = 5;
            a  = 4;
            L  = filterbanklength(tc.p.Ls, a);
            fcs = linspace(0.1, 0.9, M);
            g  = cell(1, M);
            for m = 1:M
                g{m} = blfilter('hann', 0.15, fcs(m), 'peak');
            end
            x = randn(L, 1);
            c = ufilterbank(x, g, a);

            tc.verifyEqual(size(c, 1), L/a, ...
                sprintf('ufilterbank rows: expected %d, got %d', L/a, size(c,1)));
            tc.verifyEqual(size(c, 2), M, ...
                sprintf('ufilterbank cols: expected %d, got %d', M, size(c,2)));
        end

        function testEquivalenceAudfilters(tc)
            % For audfilters (which uses uniform a), ufilterbank must match filterbank.
            [g, a_full, ~, L] = audfilters(tc.p.fs, tc.p.Ls);

            % audfilters may use non-uniform subsampling (a_full is a vector).
            % Find channels that share the same hop size and test those.
            a_scalar = a_full(1, 1);
            if ~all(a_full(:, 1) == a_scalar)
                % Non-uniform: restrict to first hop-size group
                idx = find(a_full(:, 1) == a_scalar);
                g   = g(idx);
            end
            a_vec = a_scalar * ones(numel(g), 1);
            x     = randn(L, 1);

            c_cell   = filterbank(x, g, a_vec);
            c_matrix = ufilterbank(x, g, a_scalar);

            M = numel(g);
            for m = 1:M
                err = norm(c_cell{m} - c_matrix(:, m)) / (norm(c_cell{m}) + eps);
                tc.verifyLessThan(err, 1e-12, ...
                    sprintf('audfilters band %d: ufilterbank/filterbank mismatch %.2e', m, err));
            end
        end

    end
end
